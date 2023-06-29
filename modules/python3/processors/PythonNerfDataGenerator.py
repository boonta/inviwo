# Name: PythonNeRFDataGenerator

import inviwopy as ivw
import numpy as np
from pathlib import Path
import multiprocessing as mp
import json

class PythonNeRFDataGenerator(ivw.Processor):
    """
    Documentation of PythonNeRFDataGenerator
    """
    def __init__(self, id, name):
        ivw.Processor.__init__(self, id, name)
        self.workingNetwork = ivw.app.network
        self.btnPressed = False
        self.locationValid = False
        self.imgRunner = -1
        self.rng = np.random.default_rng()
        self.genDirections = []
        self.imgDir = 'images/'
        self.genFiles = {}
        self.linkedProcessors = {}
        self.linkedProcessors['Camera'] = None
        self.linkedProcessors['Canvas'] = None
        self.cameraParameters = {}
        self.nerfData = {}
        self.cpuCount = mp.cpu_count()

        

        # self.raycasterinport = ivw.data.ImageInport("LightRaycaster")
        # self.addInport(self.raycasterinport, owner=False)

        # self.lightvolumeinport = ivw.data.VolumeInport("LightVolume")
        # self.addInport(self.lightvolumeinport, owner=False)
        # self.outport = ivw.data.ImageOutport("outport")
        # self.addOutport(self.outport, owner=False)


        self.nerfCamera = ivw.properties.CameraProperty("nerfCamera", "NeRF Camera")
        self.nerfCamera.invalidationLevel = ivw.properties.InvalidationLevel.Valid
        self.addProperty(self.nerfCamera, owner=False)

        self.nerfCameraDistance = ivw.properties.IntProperty("nerfCameraDistance", 
                                                              "Camera distance", 800, 0, 5000, 1, 
                                                              ivw.properties.InvalidationLevel.Valid)
        self.nerfCameraDistance.semantics = ivw.properties.PropertySemantics("SpinBox")
        self.addProperty(self.nerfCameraDistance, owner=False)

        self.nerfCameraDeviation = ivw.properties.IntProperty("nerfCameraDeviation", 
                                                              "Camera deviation", 20, 0, 500, 1, 
                                                              ivw.properties.InvalidationLevel.Valid)
        self.nerfCameraDeviation.semantics = ivw.properties.PropertySemantics("SpinBox")
        self.addProperty(self.nerfCameraDeviation, owner=False)

        self.nerfImageCounter = ivw.properties.IntProperty("nerfCounter", "Image counter", 1, 1, 200, 1,
                                                           ivw.properties.InvalidationLevel.Valid)
        self.nerfImageCounter.semantics = ivw.properties.PropertySemantics("SpinBox")
        self.addProperty(self.nerfImageCounter, owner=False)

        self.nerfDir = ivw.properties.DirectoryProperty("nerfDir", "NeRF save path", "", "NeRF")
        self.nerfDir.invalidationLevel = ivw.properties.InvalidationLevel.Valid
        self.addProperty(self.nerfDir, owner=False)
        self.saveNerf = ivw.properties.ButtonProperty("saveNerf", "Generate NeRF file", lambda : self.btnCallback())
        self.addProperty(self.saveNerf, owner=False)

        # This is a key component for updating the network.
        # It helps recursively evaluate the processors (specifically Canvas) for snapshot function.
        self.raycasterinport = ivw.data.ImageInport("LightRaycasterImage")  
        self.addInport(self.raycasterinport, owner=False)

    @staticmethod
    def processorInfo():
        return ivw.ProcessorInfo(
            classIdentifier="org.inviwo.PythonNeRFDataGenerator",
            displayName="Python NeRF Data Generator",
            category="Python",
            codeState=ivw.CodeState.Experimental,
            tags=ivw.Tags.PY,
            help=ivw.md2doc(PythonNeRFDataGenerator.__doc__)
        )

    def getProcessorInfo(self):
        return PythonNeRFDataGenerator.processorInfo()

    def initializeResources(self):        
        print("init")

    def process(self):
        if self.imgRunner < self.nerfImageCounter.value:
            if self.imgRunner == -1:    # Linking after this processor is created in the workspace.
                self.checkLinks()
                print(f"Process initialized.")
                pass
            if self.btnPressed and self.locationValid:  # After the buttons is pressed and the file location is valid.
                print("Process --- Generating image {}: ".format(self.imgRunner))
                print("NeRF camera is looking from {}".format(self.nerfCamera.lookFrom))
                self.capturing()
            pass

        self.updateCameraParameters()
        

    def btnCallback(self):
        self.checkLinks()
        self.btnPressed = True
        self.genDirections.clear()
        self.genFiles.clear()
        self.imgRunner = 0

        self.locationValid = self.nerfDir.value.is_dir() and str(self.nerfDir.value) != '.'
        linked = self.isCameraLinked() and self.isCanvasLinked()

        if not self.locationValid:
            print("Please locate where to save NeRF file!")
            self.btnPressed = False
            self.locationValid = False
            pass
        if not linked:
            print("Camera or Canvas is not linked.")
            self.btnPressed = False
            self.locationValid = False
            pass

        print(f"Saving NeRF file to ... {str(self.nerfDir.value)}")
        self.locationValid = True
        dist = self.nerfCameraDistance.value
        xxx = f"{str(self.nerfDir.value)}/{self.imgDir}" 
        Path(f"{str(self.nerfDir.value)}/{self.imgDir}").mkdir(parents=True, exist_ok=True)

        # random distance, theta, phi for look-for camera
        for i in range(self.nerfImageCounter.value):
            
            cdv = self.rng.uniform(low=-self.nerfCameraDeviation.value, high=self.nerfCameraDeviation.value)
            theta = np.deg2rad(self.rng.uniform(low=0, high=90))
            phi = np.deg2rad(self.rng.uniform(low=0, high=360))
            x = (dist+cdv)*np.sin(theta)*np.sin(phi)
            y = (dist+cdv)*np.cos(theta)
            z = (dist+cdv)*np.sin(theta)*np.cos(phi)
            self.genDirections.append(np.array([x,y,z]))
            self.genFiles[i] = f"{str(self.nerfDir.value)}/{self.imgDir}IMG{i:0>3d}.png"
            p = Path(self.genFiles[i])
            if p.exists():
                p.unlink()
                print(f'{p} is removed.')


    def capturing(self):
        if self.imgRunner == 0:
            self.setNeRFCamera()

        if self.imgRunner < self.nerfImageCounter.value:
            x = self.genDirections[self.imgRunner][0]
            y = self.genDirections[self.imgRunner][1]
            z = self.genDirections[self.imgRunner][2]
            self.nerfCamera.lookFrom = ivw.glm.vec3(x, y, z)    # Updating look-from parameter

            # Saving images
            print(f"Saving {self.genFiles[self.imgRunner]} : NeRF camera is looking from {self.nerfCamera.lookFrom}")
            self.linkedProcessors['Canvas'].snapshot(self.genFiles[self.imgRunner])

            # Saving camera parameters
            self.updateCameraParameters()
            self.addNeRFFrame()

            self.imgRunner = self.imgRunner+1
            if self.imgRunner == self.nerfImageCounter.value:
                self.btnPressed = False
                self.imgRunner = 0
                jsonfile = f"{str(self.nerfDir.value)}/nerffile.json"
                writingNeRF(self.nerfData, jsonfile)


    def checkLinks(self):
        # Automatically add links to Camera and Canvas, then check if all links are valid
        print(f"Adding and checking links.")
        if not self.isCameraLinked():
            self.addCameraLink()
        if not self.isCanvasLinked():
            self.addCanvasLink()

    def isCameraLinked(self):
        if self.linkedProcessors['Camera'] is None:
            print("Camera is not linked.")
            return False
        if self.workingNetwork.isLinkedBidirectional(self.nerfCamera, self.linkedProcessors['Camera'].camera):
            return True

        return False    # unhandling case

    def isCanvasLinked(self):
        if self.linkedProcessors['Canvas'] is None:
            print("Canvas is not linked.")
            return False
        if self.workingNetwork.isLinkedBidirectional(self.nerfDir, self.linkedProcessors['Canvas'].layerDir):
            return True
    
        return False    # unhandling case
    

    def addCameraLink(self):
        proc = self.workingNetwork.getProcessorByIdentifier("LightingRaycaster")
        if proc is not None:
            self.linkedProcessors['Camera'] = self.workingNetwork.LightingRaycaster
            self.workingNetwork.addLink(self.nerfCamera, self.linkedProcessors['Camera'].camera)
            self.workingNetwork.addLink(self.linkedProcessors['Camera'].camera, self.nerfCamera)
            print("Camera is now linked.")
            return proc
        print("LightingRaycaster is not found.")

    def addCanvasLink(self):
        proc = self.workingNetwork.getProcessorByIdentifier("Canvas")
        if proc is not None:
            self.linkedProcessors['Canvas'] = self.workingNetwork.Canvas
            self.workingNetwork.addLink(self.nerfDir, self.linkedProcessors['Canvas'].layerDir)
            self.workingNetwork.addLink(self.linkedProcessors['Canvas'].layerDir, self.nerfDir)
            print("Canvas is now linked.")
            return proc
        print("Canvas is not found.")

    def cameraDistance(self):
        lookto = self.nerfCamera.lookTo
        lookfrom = self.nerfCamera.lookFrom
        return ivw.glm.distance(lookfrom, lookto) 

    def updateCameraParameters(self):
        self.cameraParameters['cx'] = float(self.linkedProcessors['Canvas'].size[0]/2)
        self.cameraParameters['cy'] = float(self.linkedProcessors['Canvas'].size[1]/2)
        # self.cameraParameters['fx'] = float(self.linkedProcessors['Canvas'].size[0]/2) / np.tan(self.linkedProcessors['Camera'].camera.fov.value)
        # self.cameraParameters['fy'] = 0
        self.cameraParameters['w'] = self.linkedProcessors['Canvas'].size[0]
        self.cameraParameters['h'] = self.linkedProcessors['Canvas'].size[1]
        self.cameraParameters['fov'] = self.linkedProcessors['Camera'].camera.fov.value
        self.cameraParameters['skew'] = 0
        self.cameraParameters['ar'] = self.linkedProcessors['Camera'].camera.aspectRatio
        self.cameraParameters['near'] = self.linkedProcessors['Camera'].camera.nearPlane
        self.cameraParameters['far'] = self.linkedProcessors['Camera'].camera.farPlane
        self.cameraParameters['viewMatrix'] = np.array(self.linkedProcessors['Camera'].camera.viewMatrix)   # corresponding to the numpy array used in NeRF
        self.cameraParameters['projectionMatrix'] = np.array(self.linkedProcessors['Camera'].camera.projectionMatrix)

        # print(self.cameraParameters['projectionMatrix'])
        # print(type(self.cameraParameters['viewMatrix']))
        # print(self.cameraParameters['viewMatrix'].tolist())
        # toRowMajor(self.linkedProcessors['Camera'].camera.viewMatrix)

        pass

    def setNeRFCamera(self):
        self.nerfData['camera_model'] = "OPENGL_INVIWO"
        self.nerfData['w'] = self.cameraParameters['w']
        self.nerfData['h'] = self.cameraParameters['h']
        self.nerfData['cx'] = self.cameraParameters['cx']
        self.nerfData['cy'] = self.cameraParameters['cy']
        self.nerfData['fov'] = self.cameraParameters['fov']
        self.nerfData['ar'] = self.cameraParameters['ar']
        self.nerfData['near'] = self.cameraParameters['near']
        self.nerfData['far'] = self.cameraParameters['far']


    def addNeRFFrame(self):
        newFrame = {}
        newFrame['imgPath'] = f"{self.imgDir}{Path(self.genFiles[self.imgRunner]).name}"
        newFrame['viewMatrix'] = self.cameraParameters['viewMatrix'].tolist()

        if 'frames' not in self.nerfData:
            self.nerfData['frames'] = []
        self.nerfData['frames'].append(newFrame)
        

def distance(A,B):
    dx = A[0]-B[0]
    dy = A[1]-B[1]
    dz = A[2]-B[2]
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def writingNeRF(jsonData, jsonFile):
    print("writing NeRF.")
    jsonObj = json.dumps(jsonData, indent=4)
    try:
        with open(jsonFile, "w") as output:
            output.write(jsonObj)
    except:
        print(f'Error writing NeRF to {jsonFile}')
    pass

