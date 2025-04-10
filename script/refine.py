import os,sys

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import simtk.openmm as mm
import subprocess

def woutpdb( infile, outfile):
    lines = open(infile).readlines()
    nums=[]
    for aline in lines:
        if aline.startswith('ATOM'):
            num = aline.split()[5]
            nums.append(num)
    nums.sort()
    olines=[]
    for aline in lines:
        if aline.startswith('ATOM'):
            atom = aline.split()[2]
            num = int(aline.split()[5])

            theline = list(aline)
            if num == nums[0]:
                theline[18:20] = theline[19:20] + ['5']
            elif num == nums[-1]:
                theline[18:20] = theline[19:20] + ['3']
            if not (("P" in atom and num == 1)   or ("H" in atom ) ):
                olines.append(''.join(theline))
    wfile = open(outfile,'w')
    wfile.write(''.join(olines))
    wfile.close()
            
def woutpdb2( infile, outfile):
    lines = open(infile).readlines()
    olines = []
    for aline in lines:
        if aline.startswith('ATOM') and 'H' not in aline:
            olines.append(aline)
    wfile = open(outfile,'w')
    wfile.write(''.join(olines))
    wfile.close()   



def opt(inpdb,outpdb,steps):

    # https://openmm.org/
    pdb = PDBFile(inpdb)
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller.addHydrogens(forcefield)
    modeller.addSolvent(forcefield, padding=1 * nanometer)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1 * nanometer,
                                        constraints=HBonds)
    if False:
        #restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        restraint = CustomExternalForce('k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
        system.addForce(restraint)
        restraint.addGlobalParameter('k', 100.0*kilojoules_per_mole/nanometer)
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')
        for atom in pdb.topology.atoms():
            # print(atom.name)
            if atom.name == 'P':
                # print('added')
                restraint.addParticle(atom.index, pdb.positions[atom.index])
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    simulation.minimizeEnergy(maxIterations=steps)
    position = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(simulation.topology, position, open(outpdb, 'w'))



inpdb = sys.argv[1]
outpdb = sys.argv[2]
count = len(open(inpdb).readlines())
steps=(int(count/20.0)) * 1
if len(sys.argv) == 4:
    steps=int(float(sys.argv[3]) * steps)
steps=max(steps,10)
steps=min(steps,1000)
print('steps:',steps)
woutpdb( inpdb, outpdb+'amber_tmp.pdb')
opt(outpdb+'amber_tmp.pdb',outpdb+'amber_tmp2.pdb',steps)
woutpdb2( outpdb+'amber_tmp2.pdb', outpdb)
try:
    os.remove(outpdb+'amber_tmp.pdb')
    os.remove(outpdb+'amber_tmp2.pdb')
except:
    pass