# Ligand:  NPA (4-nitrophenyl acetate)

import os, sys, time
from openmm import app, unit, Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from openmmml import MLPotential
from parmed import load_file
from parmed.openmm import RestartReporter

# Load AMBER files
amber_sys = load_file("prmtop.parm7", "restart.rst7")
ncrst=app.amberinpcrdfile.AmberInpcrdFile("restart.rst7")

# Simulation parameters
nsteps = 20000
timestep = 1.0 * unit.femtoseconds
temperature = 310 * unit.kelvin
pressure = 1 * unit.atmosphere

# Create base MM system
system = amber_sys.createSystem (
    nonbondedMethod = app.PME,
    ewaldErrorTolerance = 0.0004,
    nonbondedCutoff = 8.0 * unit.angstroms,
    constraints = app.HBonds,
    removeCMMotion = True
)

# Identify NPA atoms
npa_atoms = [ ] 
for atom in amber_sys.topology.atoms():
    if atom.residue.name == "NPA":
        npa_atoms.append(atom.index)

# Create ANI2X mixed system
potential = MLPotential('ani2x')

system = potential.createMixedSystem(
    amber_sys.topology,
    system,
    npa_atoms,
    removeConstraints=False
)

integrator = LangevinMiddleIntegrator(temperature, 1.0 / unit.picoseconds, timestep)
barostat = MonteCarloBarostat(pressure, temperature, 50)
system.addForce(barostat)

platform = Platform.getPlatformByName("CUDA")
print("Platform:", platform.getName())
properties = dict(
    CudaPrecision = "mixed",
    UseCpuPme = 'false',
    DeterministicForces = "false",
    DeviceIndex = os.environ["CUDA_VISIBLE_DEVICES"],
)

sim = app.Simulation(amber_sys.topology, system, integrator, platform, properties)
sim.context.setPositions(amber_sys.positions)
#sim.context.setVelocitiesToTemperature(310)
sim.context.setVelocities(ncrst.velocities)

sim.reporters.append(app.StateDataReporter(
    sys.stdout, 
    1000,
    step = True, 
    potentialEnergy = True, 
    kineticEnergy = True,
    totalEnergy = True, 
    temperature = True, 
    volume = True
    )
)

sim.reporters.append(app.DCDReporter("trajectory.dcd", 50000))
sim.reporters.append(RestartReporter("restart.nc", 50000, netcdf = True))

print("\nRunning dynamics...")
start = time.time()

sim.step(nsteps)

elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} sec")

# Benchmark
simulated_ns = (nsteps * timestep).value_in_unit(unit.nanoseconds)
ns_per_day = simulated_ns / (elapsed / 86400)

print(f"Performance: {ns_per_day:.2f} ns/day")

