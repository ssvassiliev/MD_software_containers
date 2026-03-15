import openmm as mm
import openmm.app as app
from openmm.unit import *
import os, time, netCDF4
from parmed import load_file
from parmed.openmm import StateDataReporter, NetCDFReporter

nsteps=20000
dt=1.0

amber_sys=load_file("prmtop.parm7", "restart.rst7")
ncrst=app.amberinpcrdfile.AmberInpcrdFile("restart.rst7")

system=amber_sys.createSystem(
            nonbondedMethod=app.PME, 
            ewaldErrorTolerance=0.0005,
            nonbondedCutoff=8.0*angstroms,
            constraints=app.HBonds,
            removeCMMotion = True,
)

integrator = mm.LangevinMiddleIntegrator(300*kelvin, 1.0/picoseconds, dt*femtoseconds,)
barostat = mm.MonteCarloBarostat(1.0*atmosphere, 300.0*kelvin, 50)
system.addForce(barostat)

platform = mm.Platform.getPlatformByName("HIP")

prop = dict(
    Precision="mixed", 
    UseCpuPme='false', 
    DeterministicForces="false", 
    DeviceIndex=os.environ["SLURM_JOB_GPUS"],
   )

sim = app.Simulation(amber_sys.topology, system, integrator, platform, prop)
sim.context.setPositions(amber_sys.positions)
sim.context.setVelocities(ncrst.velocities)

nonbonded = next(f for f in system.getForces()
                 if isinstance(f, mm.NonbondedForce))
params = nonbonded.getPMEParametersInContext(sim.context)
print("Nonbonded settings:")
print("  Cutoff:", nonbonded.getCutoffDistance())
print("  Switching enabled:", nonbonded.getUseSwitchingFunction())
print("  PME alpha:", params[0])
print("  PME grid:", params[1], params[2], params[3])
print("  Ewald tolerance:", nonbonded.getEwaldErrorTolerance())
print("  Method:", nonbonded.getNonbondedMethod())

print("\nAll system forces:")
for i, force in enumerate(system.getForces()):
    print(f"  {i}: {force.__class__.__name__}")
    if isinstance(force, mm.MonteCarloBarostat):
        print("    Frequency:", force.getFrequency())
        print("    Pressure:", force.getDefaultPressure())
        print("    Temperature:", force.getDefaultTemperature())

print("Warming up")
sim.step(100)  # Compile all kernels
sim.context.setTime(0)  # Reset time
sim.currentStep = 0 

sim.reporters.append(
        StateDataReporter(
            sys.stdout,
            1000,
            step=True,
            time=False,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            volume=True
        )
)

sim.reporters.append(
        NetCDFReporter(
            "trajectory.nc",
            50000,
            crds=True
        )
)

print("Running dynamics")
start = time.time()
sim.step(nsteps)
elapsed=time.time() - start
benchmark_time = 0.0864 * nsteps * dt / elapsed
print(f"Elapsed time: {elapsed} sec\nBenchmark time: {benchmark_time} ns/day, ", end="")

