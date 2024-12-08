import numpy as np 
import math



def fractional_converge_test(Fnew,Fold):
    return 2.0*math.fabs(Fnew-Fold)<=1e-4*  (math.fabs(Fnew)+math.fabs(Fold)+1e-10)

def absolute_converge_test(Fnew,Fold):
    return math.fabs(Fnew-Fold) <= 0.01

class lbfgs_iteration_data:
    def __init__(self):
        self.alpha=0
        self.s=0
        self.y=0
        self.ys=0

class func_1d:
    #cdef public object _starting_point,_search_direction,_eval_point,_dE_dvars,_func,_dfun
    #cdef public int _eval_count,_size,_deriv_count
    def __init__(self,size,start,search_direction,func,dfunc):
        self._size=size
        self._starting_point=start
        self._search_direction=search_direction
        self._eval_point=np.zeros_like(start)
        self._dE_dvars=np.zeros_like(start)
        self._func=func
        self._dfun=dfunc
        self._eval_count=0
        self._deriv_count=0
    def eva(self,displacement):
        self._eval_count+=1
        self._eval_point=self._starting_point+ (displacement * self._search_direction)
        return self._func(self._eval_point)

    def dfunc(self,displacement):
        self._deriv_count+=1
        self._eval_point=self._starting_point+ (displacement * self._search_direction)
        self._dE_dvars=self._dfun(self._eval_point)
        return np.dot(self._dE_dvars,self._search_direction)

class ArmijoLineMinimization:
    #cdef public  object  _func,_dunc,nonmonotone
    #cdef public  int dim,_num_linemin_calls,_num_calls
    #cdef public  double _deriv_sum,max_step_limit,_tolerance,_last_accepted_step,_func_to_beat


    def __init__(self,func,dfunc,nonmonotone,dim,max_step_limit):
        self._func=func
        self._dunc=dfunc
        self.nonmonotone=nonmonotone
        self.dim=dim
        self.max_step_limit=max_step_limit
        self._num_linemin_calls=0
        self._last_accepted_step=1.0
        self._func_to_beat=0.0
        self._deriv_sum=0.0
        self._tolerance=0.1
        self._num_calls=0

    def eva(self,current_position,search_direction):
        self._num_linemin_calls+=1
        FACTOR=0.5
        
        problem_size = self.dim
        this_line_func = func_1d(problem_size,current_position,search_direction,self._func,self._dunc)
        derivmax = np.max( np.abs(search_direction)  )
        if derivmax < 0.0001:
            #print('maxnorm is ',derivmax,'do not move')
            return this_line_func.eva(0.0),current_position,search_direction
        init_step = self._last_accepted_step / FACTOR
        if init_step > self.max_step_limit:
            init_step = self.max_step_limit  
        final_value = self.Armijo(init_step, this_line_func) 
        #print('line search done, last_accepted_step:',self._last_accepted_step)     
        search_direction = search_direction * self._last_accepted_step
        current_position = current_position + search_direction

        return final_value,current_position,search_direction
    def Armijo(self,init_step,func_eval):
        FACTOR = 0.5
        SIGMA = 0.1
        SIGMA2 = 0.8
        MINSTEP = 1e-9
        func_value = func_eval.eva( init_step )
        self._num_calls+=1
        self._last_accepted_step = init_step
        if func_value < self._func_to_beat + init_step * SIGMA2 * self._deriv_sum:
            #print("func_value, '<',init_step * SIGMA2 * self._deriv_sum",func_value, '<',init_step * SIGMA2 * self._deriv_sum)
            test_step = init_step/FACTOR
            test_func_value = func_eval.eva( test_step )
            self._num_calls+=1
            if test_func_value < func_value:
                #print('test func good',test_func_value)
                self._last_accepted_step = test_step
                return test_func_value
            #print('test func not good',test_func_value)
            return func_value
        far_step = init_step
        while func_value > self._func_to_beat + init_step*SIGMA*self._deriv_sum:
            if (init_step <= 1e-5 * far_step) or ((init_step < MINSTEP) and (func_value >= self._func_to_beat)):
                print('Abort if function value is unlikely to improve. Inaccurate G!',func_value , self._func_to_beat)
                #test_step = ( func_value - self._func_to_beat ) / init_step
                self._last_accepted_step = 0.0
                return self._func_to_beat
            init_step = init_step* FACTOR*FACTOR
            func_value = func_eval.eva( init_step );
            self._num_calls+=1
        self._last_accepted_step = init_step
        if init_step < 0.0:
            test_step = -self._deriv_sum*init_step*init_step/(2*(func_value - self._func_to_beat - init_step * self._deriv_sum))
            if ( test_step > 1e-3*far_step and  test_step < far_step ):
                test_func_value = func_eval.eva(test_step)
                self._num_calls+=1
                if test_func_value < func_value:
                    self._last_accepted_step = test_step
                    func_value = test_func_value
        return func_value

def update(bound,j,M,M_alpha,M_s,D,M_y,M_ys):
    for pts in range(bound):
        j=j-1
        if j<=0:
            j=M
        M_alpha[j-1] = np.dot(M_s[j-1] , D)
        M_alpha[j-1] = M_alpha[j-1] / M_ys[j-1]
        D = D - M_alpha[j-1] * M_y[j-1]
    for pts in range(bound):
        beta = np.dot(M_y[j-1],D)
        beta = beta / M_ys[j-1]
        D = D + (M_alpha[j-1] - beta) * M_s[j-1]
        j = j+1
        if j > M:
            j=1
    return bound,j,M,M_alpha,M_s,D,M_y,M_ys

                    
class lbfgs:
    #cdef object func,dfunc
    #cdef double gmax_cutoff_for_convergence
    def __init__(self,func,dfun):
        self.func=func
        self.dfunc=dfun
        self.gmax_cutoff_for_convergence = 1e-8
    
    def run(self,x_,M,converge_test,line_min,ITMAX,func,dfunc,gmax_cutoff_for_convergence):
        FRET =0
        x = x_+0.0
        N=len(x)

        PAST =1
        if line_min.nonmonotone:
            PAST=3
        EPS = 1e-5

        K = 1
        XP = np.zeros_like(x)
        Xtemp = np.zeros_like(x)
        G = np.zeros_like(x)
        GP = np.zeros_like(x)
        Gtemp = np.zeros_like(x)
        D = np.zeros_like(x)
        W = np.zeros_like(x)

        CURPOS = 1
        M_alpha=np.zeros(M)
        M_s=np.zeros([M,N])
        M_y=np.zeros([M,N])
        M_ys=np.zeros(M)

        pf = np.zeros(PAST)
        func_memory_filled = 1
        prior_func_value = func(x)
        pf[0]=FRET=prior_func_value
        G=dfunc(x)

        invdnorm =0.0
        D=-G
        if line_min.nonmonotone:
            line_min._last_accepted_step=0.005
        last_step_good=True
        for ITER in range(ITMAX):
            if last_step_good:
                XP = x+0.0
                GP = G+0.0
            line_min._deriv_sum = 0.0
            Gmax = np.max(np.abs(G))
            line_min._deriv_sum = np.dot(D,G)
            line_min._func_to_beat = np.max(pf[:func_memory_filled])
            if line_min._deriv_sum > -EPS:
                #print('gradient the same direction, reset1',line_min._deriv_sum)
                line_min._deriv_sum=0.0
                D[D*G >=0]*=-1.0
                line_min._deriv_sum = np.dot(D,G)
                Gmax = np.max(np.abs(G))
            if line_min._deriv_sum > -EPS:
                #print('gradient the same direction, reset2',line_min._deriv_sum)
                D=-1.0*G
                line_min._deriv_sum=np.dot(D,G)
                if math.sqrt( - line_min._deriv_sum) > 1e-6:
                    func_memory_filled = 1
                    line_min._func_to_beat = pf[0] =prior_func_value =FRET
            FRET, x,D = line_min.eva(x,D)
            if converge_test(FRET,prior_func_value) or line_min._last_accepted_step ==0:
                if Gmax<=gmax_cutoff_for_convergence :
                    #print('max abs gradient is ',Gmax,'done!')
                    return x
                else:
                    if line_min._last_accepted_step ==0:
                        #print('max abs gradient is ',Gmax,'too large, reset!')
                        CURPOS = 1
                        K =1
                        line_min._deriv_sum=0.0
                        D=-1.0*G
                        line_min._deriv_sum=np.dot(D,G)
                        if line_min._deriv_sum <= -EPS:
                            invdnorm = 1.0/math.sqrt( -line_min._deriv_sum )
                            line_min._last_accepted_step = invdnorm
                            func_memory_filled = 1
                            prior_func_value = FRET
                            pf[0] = prior_func_value
                            FRET,x,D = line_min.eva(x,D)
                        else:
                            return x
                        if line_min._last_accepted_step ==0:
                            print('Line search failed even after resetting Hessian;')
                            return x
            prior_func_value = FRET
            #print('update memory, learning rate:',line_min._last_accepted_step)
            if func_memory_filled < PAST:
                func_memory_filled+=1
            else:
                tmppf=pf[1:]+0.0
                pf[:-1]=tmppf
            pf[func_memory_filled-1]=prior_func_value
            if ITER%50==0:
                print('At iterate',ITER,'   f=',min(pf),'  max Grad=',Gmax)
                if np.abs(pf[-1]-pf[-2])<1e-3:
                    return x
            GP = G+0.0

            G = dfunc(x)
            line_min._deriv_sum = 0.0
            deriv_new = 0.0
            line_min._deriv_sum = np.dot(D,GP)
            deriv_new = np.dot(D,G)

            ys,yy=0,0
            Xtemp = x - XP
            Gtemp = G - GP
            ys = np.dot(Gtemp,Xtemp)
            yy = np.dot(Gtemp,Gtemp)

            if math.fabs(ys) < 1e-6:
                last_step_good = False
            else:
                last_step_good = True;
                M_s[CURPOS-1] = Xtemp
                M_y[CURPOS-1] = Gtemp
                M_ys[CURPOS-1] =ys
                K+=1
                CURPOS+=1
                if CURPOS >M:
                    CURPOS = 1

            bound = min(M,K-1)
            D=-1.0*G
            j = CURPOS + 0 

            bound,j,M,M_alpha,M_s,D,M_y,M_ys=update(bound,j,M,M_alpha,M_s,D,M_y,M_ys)
            # for pts in range(bound):
            #     j=j-1
            #     if j<=0:
            #         j=M
            #     M_alpha[j-1] = np.dot(M_s[j-1] , D)
            #     M_alpha[j-1] = M_alpha[j-1] / M_ys[j-1]
            #     D = D - M_alpha[j-1] * M_y[j-1]
            # for pts in range(bound):
            #     beta = np.dot(M_y[j-1],D)
            #     beta = beta / M_ys[j-1]
            #     D = D + (M_alpha[j-1] - beta) * M_s[j-1]
            #     j = j+1
            #     if j > M:
            #         j=1


        return x



































