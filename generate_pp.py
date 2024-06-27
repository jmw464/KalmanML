#!/usr/bin/env python3

import pathlib, acts, acts.examples, os
from acts.examples.simulation import addPythia8, MomentumConfig, EtaConfig, ParticleConfig, addFatras, addDigitization, ParticleSelectorConfig
from acts.examples.reconstruction import addSeeding, addCKFTracks, TrackSelectorConfig, TruthSeedRanges, CkfConfig
from acts.examples.odd import getOpenDataDetector
import sys

# BEGIN INPUTS

chi2cut = 15.0 # Default: 15

actsdir = "/home/max_zhao/acts/"
datadir = "/global/cfs/cdirs/atlas/max_zhao/mlkf/trackml/test_events/eval_test/"
nevents = 5
cms_energy = 14 #TeV
npileup = 200
detector = "generic"
rnd_seed = 1234

# END INPUTS

u = acts.UnitConstants
outputDir = datadir+"ttbar"+str(npileup)+"_"+str(nevents)+"/"
#outputDirCsv = outputDir+"csv/"

if not os.path.exists(datadir): os.mkdir(datadir)
if not os.path.exists(outputDir): os.mkdir(outputDir)
#if not os.path.exists(outputDirCsv): os.mkdir(outputDirCsv)

if detector == "generic":
    DigiConfig = actsdir+"/Examples/Algorithms/Digitization/share/default-geometric-config-generic.json"
    SeedingSel = actsdir+"/Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json"
    detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()
elif detector == "odd":
    geoDir = actsdir+"thirdparty/OpenDataDetector/"
    MaterialMap = geoDir+"data/odd-material-maps.root"
    DigiConfig = geoDir+"config/odd-digi-geometric-config.json"
    SeedingSel = geoDir+"config/odd-seeding-config.json"
    MaterialDeco = acts.IMaterialDecorator.fromFile(MaterialMap)
    detector, trackingGeometry, decorators = getOpenDataDetector(geoDir, mdecorator=MaterialDeco)
else:
    print("Choose odd or generic detector")

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=rnd_seed)

s = acts.examples.Sequencer(events=nevents, numThreads=1, outputDir=str(outputDir))

addPythia8(
    s,
    cmsEnergy=cms_energy*u.TeV,
    npileup=npileup,
    hardProcess=["Top:qqbar2ttbar=on"],
    vtxGen=acts.examples.GaussianVertexGenerator(stddev=acts.Vector4(0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns), mean=acts.Vector4(0, 0, 0, 0)),
    rnd=rnd,
    outputDirRoot=outputDir,
#    outputDirCsv=outputDirCsv,
)

addFatras(
    s,
    trackingGeometry,
    field,
    preSelectParticles=ParticleSelectorConfig(eta=(-3.0, 3.0), pt=(150 * u.MeV, None), removeNeutral=True),
    rnd=rnd,
    outputDirRoot=outputDir,
#    outputDirCsv=outputDirCsv,
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=DigiConfig,
    rnd=rnd,
    outputDirRoot=outputDir,
#    outputDirCsv=outputDirCsv,
)

addSeeding(
    s,
    trackingGeometry,
    field,
    TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-3.0, 3.0), nHits=(9, None)),
    geoSelectionConfigFile=SeedingSel,
    outputDirRoot=outputDir,
)

addCKFTracks(
    s,
    trackingGeometry,
    field,
    TrackSelectorConfig(
        pt=(1.0 * u.GeV, None),
        absEta=(None, 3.0),
        loc0=(-4.0 * u.mm, 4.0 * u.mm),
        nMeasurementsMin=6,
    ),
    CkfConfig(
        chi2CutOff=chi2cut
    ),
    outputDirRoot=outputDir,
    writeCovMat=True
#    outputDirCsv=outputDirCsv,
)

s.run()
