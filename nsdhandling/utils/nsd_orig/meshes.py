import numpy as np
from scipy import interpolate
from mayavi import mlab
import igl
import nibabel as nib

class Face:
    def __init__(self,vertex_ids=None):
        """
        A class for faces of triangulation
        :param vertex_ids: 3 ints for vertex ids
        """
        self.vertex_ids = vertex_ids

class Vertex:
    def __init__(self,coords=None):
        """
        A class for vertices making up a triangulation
        :param coords: 3D coordinates of each vertex
        :param faces: Id's of faces that the vertex belongs to
        """
        self.coords = np.asarray(coords)
        self.faces = []

class Surface:
    def __init__(self):
        """
        Class to store surface surface
        """
        self.subject=[]
        self.hemi=[]
        self.vertices= []
        self.faces = []
        self.volume_info = []
        self.gaussian_curvature=[]
        self.freesurfer_curvature=[]
        self.scalar = []
        self.vector = []
        self.tensor = []
        self.points = []
        self.have_points=0

    def getSurf(self,path=None, hemi=None, surf=None):
        """
        Gets Freesurfer mesh using loadSurf
        :param subject: Subject id
        :param hemi: Which hemisphere
        :param surf: Which surface (pial, white, etc.)
        :param kwargs:
        :return: Will store new mesh in self.vertices and self.faces of same class
        """
        #self.subject=subject
        self.path=path
        self.hemi=hemi
        coords, faces  = self.loadSurf(path,hemi,surf)
        self.vertices=[Vertex(acoord) for acoord in coords]
        f=0 #there is probably a better way to do this
        for aface in faces:
            self.faces.append(Face(vertex_ids=aface))
            for node in range(3):
                self.vertices[aface[node]].faces.append(f)
            f=f+1
        del coords
        del faces

    def loadSurf(self,path, hemi, surf):
        return nib.freesurfer.io.read_geometry(path+'/'+hemi+'.'+surf)

    def getSubmesh(self, surface, index): #filter out a submesh based on a label for vertices
        """
        Gets submesh based on freesurfer labels, submesh is a new mesh with new indices for faces, vertices
        :param surface: Mesh to extract submesh from, usually a hemisphere
        :param index: Freesurfer label of region to extract
        :return: Will store new mesh in self.vertices and self.faces of same class
        """
        if surface.aparc is None:
            raise ValueError("Aparc not loaded")
        print("fetching submesh as a new mesh")
        inds = np.where(surface.aparc.labels == index)
        inds = inds[0][:]
        self.vertices = [Vertex() for ind in inds]
        f=0
        for i in range(inds.shape[0]):
            ind=inds[i]
            self.vertices[i].coords = surface.vertices[ind].coords
            faceids= surface.vertices[ind].faces
            for faceid in faceids:
                vert_ids_in_face = surface.faces[faceid].vertex_ids
                if (set(vert_ids_in_face) & set(inds)) == set(vert_ids_in_face):
                    ids_in_face = np.array([0, 0, 0])
                    for k in range(3):
                        ind_in_face=np.where(inds == vert_ids_in_face[k])
                        ind_in_face=ind_in_face[0][0]
                        self.vertices[ind_in_face].faces.append(f)
                        if vert_ids_in_face[k] != ind:
                            surface.vertices[vert_ids_in_face[k]].faces.remove(faceid)
                        ids_in_face[k]=ind_in_face
                    f=f+1
                    self.faces.append(Face(vertex_ids=ids_in_face))
        self.getGaussCurv()

    def project(self, volume=None, header=None):
        """
        Project data onto surface
        :param volume: Volume class for gridded data
        :param header: Header from original.mgz to create correct xfm
        :return: Fills out self.scalar, self.vector etc. in same class
        """
        #vol should be nibabel nifti object and mesh is surface we want to project onto
        if volume is None:
            raise ValueError("no volume provided")
        if header is None:
            raise ValueError("no orig.mgz header provided, need this for transforms")

        #we will bring vertex coordinates to voxel space and then interpolate
        #inverse of tkrvox2ras will bring us to voxel space of orig
        #sfrom of orig brings us to world coordinates
        #inverse of sform of vol brings the coordinates to voxel space of vol where we can interpolate on grid

        Torig=np.asmatrix(header.get_vox2ras_tkr())
        Norig=np.asmatrix(header.get_vox2ras())
        sform=np.asmatrix(volume.vol.get_sform())

        xfm=np.matmul(Norig, np.linalg.inv(Torig))
        xfm=np.matmul(np.linalg.inv(sform), xfm)

        shape=volume.vol.shape

        points=[]
        if self.have_points == 0:
            for vertex in self.vertices:
                #self.vertices.append(Vertex(coords=vertex.coords))
                #point=np.asarray(vertex.coords)
                point = vertex.coords
                point=np.append(point,1)
                point=np.asarray(xfm @ point) #this line is so bloated has to be better way
                points.append(point[0,0:3])
                self.points=points
                self.have_points=1

        #TODO have to put something for other stuff too like scalars, etc.
        if len(shape) > 3:
            print(len(shape))
            if shape[3] > 1:
                temp = []
                self.scalar=[]
                for j in range(shape[3]):
                    print('j',j)
                    #temp.append(volume.interpolator[j](self.points))
                    #self.scalar.append(temp[0][:])
                    self.scalar.append(volume.interpolator[j](self.points))
                #self.vector = np.column_stack((temp[0], temp[1], temp[2]))

            if shape[3] == 1:
                temp = []
                temp.append(volume.interpolator(self.points))
                self.scalar = temp[0][:]
        else:
            temp = []
            #temp.append(volume.interpolator(self.points))
            #self.scalar = temp[0][:]
            self.scalar = volume.interpolator(self.points)


class Projection: #this is to project volume data onto mesh
    def __init__(self):
        """
        Class for projecting data onto surfaces
        """
        self.vertices=[]
        self.scalar=[]
        self.vector=[]
        self.tensor=[]
        self.points=[]

    def project(self, volume=None, mesh=None, header=None):
        """
        Project data onto surface
        :param volume: Volume class for gridded data
        :param mesh: Mesh of surface to project data on
        :param header: Header from original.mgz to create correct xfm
        :return: Fills out self.scalar, self.vector etc. in same class
        """
        #vol should be nibabel nifti object and mesh is surface we want to project onto
        if volume is None:
            raise ValueError("no volume provided")
        if mesh is None:
            raise ValueError("no mesh to be projected on provided")
        if header is None:
            raise ValueError("no orig.mgz header provided, need this for transforms")

        #we will bring vertex coordinates to voxel space and then interpolate
        #inverse of tkrvox2ras will bring us to voxel space of orig
        #sfrom of orig brings us to world coordinates
        #inverse of sform of vol brings the coordinates to voxel space of vol where we can interpolate on grid

        Torig=np.asmatrix(header.get_vox2ras_tkr())
        Norig=np.asmatrix(header.get_vox2ras())
        sform=np.asmatrix(volume.vol.get_sform())

        xfm=np.matmul(Norig, np.linalg.inv(Torig))
        xfm=np.matmul(np.linalg.inv(sform), xfm)

        shape=volume.vol.shape

        points=[]
        for vertex in mesh.vertices:
            #self.vertices.append(Vertex(coords=vertex.coords))
            #point=np.asarray(vertex.coords)
            point = vertex.coords
            point=np.append(point,1)
            point=np.asarray(xfm @ point) #this line is so bloated has to be better way
            self.points.append(point[0,0:3])

        #TODO have to put something for other stuff too like scalars, etc.
        if  len(shape) > 3:
            if shape[3] > 1:
                temp=[]
                for j in range(shape[3]):
                    print('Projecting vol:', j)
                    temp.append(volume.interpolator[j](self.points))
                self.vector = np.column_stack((temp[0],temp[1],temp[2]))
            if shape[3] == 1:
                temp = []
                temp.append(volume.interpolator(self.points))
                self.scalar = temp[0][:]
        else:
            temp = []
            temp.append(volume.interpolator(self.points))
            self.scalar = temp[0][:]

class Volume():
    def __init__(self):
        """
        Class for storing gridded volume data
        """
        self.vol = []
        self.interpExists = 0
        self.interpolator = []

    def getVolume(self, path=None):
        """
        Gets volume data
        :param filename: Path of volume file
        :return:
        """
        self.vol=nib.load(path)

    def makeInterpolator(self):
        """
        Makes a linear interpolator
        :return: Fills out self. interpolator and sets self.interpExists = 1 after interpolator is calculated
        """
        shape = self.vol.shape
        print(shape)
        img = self.vol.get_data()
        #TODO other shapes like scalars most impot
        if  len(shape) > 3:
            if shape[3] > 1:
                i = np.linspace(0, shape[0] - 1, num=shape[0])
                j = np.linspace(0, shape[1] - 1, num=shape[1])
                k = np.linspace(0, shape[2] - 1, num=shape[2])
                self.interpolator = [interpolate.RegularGridInterpolator((i, j, k), img[:, :, :, f],method='linear')
                                     for f in range(
                    shape[3])]
                self.interpExists=1
            if shape[3]==1:
                i = np.linspace(0, shape[0] - 1, num=shape[0])
                j = np.linspace(0, shape[1] - 1, num=shape[1])
                k = np.linspace(0, shape[2] - 1, num=shape[2])
                self.interpolator = interpolate.RegularGridInterpolator((i, j, k), img[:, :, :,0],method='nearest')
                self.interpExists = 1
        else:
            i = np.linspace(0, shape[0] - 1, num=shape[0])
            j = np.linspace(0, shape[1] - 1, num=shape[1])
            k = np.linspace(0, shape[2] - 1, num=shape[2])
            self.interpolator = interpolate.RegularGridInterpolator((i, j, k), img[:, :, :],method='nearest')
            self.interpExists = 1



class Vision:
    def __init__(self):
        self.mesh=[]
        self.scalar=[]
        self.vector=[]
        self.x=[]
        self.y=[]
        self.z=[]
        self.triangles=[]
        self.vector_added=0
        self.scalar_added=0


    def processMesh(self, mesh=None):
        self.mesh=mesh
        x=[]
        y=[]
        z=[]
        triangles=[]
        for vertex in self.mesh.vertices:
            x.append(vertex.coords[0])
            y.append(vertex.coords[1])
            z.append(vertex.coords[2])
        triangles = np.row_stack([face.vertex_ids for face in self.mesh.faces])
        self.x=x
        self.y=y
        self.z=z
        self.triangles=triangles
        #mlab.triangular_mesh(self.x,self.y,self.z,self.triangles)
        #mlab.triangular_mesh(x, y, z, triangles)

    def addVector(self, vector=None): #for now this will take vector fields from Projection class
        tempVector=np.row_stack([thisVector for thisVector in vector])
        self.vector=tempVector
        self.vector_added=1

    def addScalar(self, scalar=None):
        self.scalar=scalar
        self.scalar_added=1

    def show(self):
        if self.scalar_added==1:
            mlab.triangular_mesh(self.x,self.y,self.z,self.triangles, scalars=self.scalar)
        else:
            mlab.triangular_mesh(self.x, self.y, self.z, self.triangles)
        if self.vector_added==1:
            mlab.quiver3d(self.x, self.y, self.z, self.vector[:, 0], self.vector[:, 1], self.vector[:, 2])
        mlab.show()