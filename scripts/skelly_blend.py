import sys
import site
import bpy
import numpy as np
import bmesh
import os
import pickle

site_package_dir = site.getusersitepackages()
if not site_package_dir in sys.path:
    sys.path.append(site_package_dir)

try:
    import msgpack, toml
except ImportError:
    import subprocess
    PYTHON = sys.executable

    # upgrade pip
    subprocess.call([PYTHON, "-m", "ensurepip"])
    subprocess.call([PYTHON, "-m", "pip", "install", "--upgrade", "pip"])

    # install required packages
    subprocess.call([PYTHON, "-m", "pip", "install", "msgpack"])
    subprocess.call([PYTHON, "-m", "pip", "install", "toml"])

    import msgpack, toml


def _create_sphere(position : np.array, radius : float, name : str, material : str, half=False):
    mesh = bpy.data.meshes.new(name)
    sphere = bpy.data.objects.new(name, mesh)
    sphere.data.materials.append(bpy.data.materials[material])
    bpy.context.collection.objects.link(sphere)

    bpy.context.view_layer.objects.active = sphere
    sphere.select_set(True)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm,
                              u_segments=64,
                              v_segments=32,
                              diameter=radius)

    if half:
        modifier = sphere.modifiers.new(name="Solidify", type="SOLIDIFY")
        modifier.thickness = 0.1
        modifier.offset = 1.0

        for vert in bm.verts:
            if vert.co[1] > 0.0001 * radius:
                bm.verts.remove(vert)
    bm.to_mesh(mesh)
    bpy.ops.object.shade_smooth()
    bm.free()
    return sphere

def place_shell(radius, half=True):
    sphere = _create_sphere(np.zeros(3), radius, 'Shell', 'ShellMaterial', half)

def create_body(position : np.array, radius : float):
    sphere = _create_sphere(position, radius, 'Body', 'BodyMaterial', False)
    bpy.context.collection.objects.unlink(sphere)
    bpy.data.collections['Bodies'].objects.link(sphere)

def init_materials():
    if not 'FiberMaterial' in bpy.data.materials:
        new_shader("FiberMaterial", "glossy", 0.087, 0.381, 1.0)
        bpy.data.materials["FiberMaterial"].roughness = 0.798
    if not 'ShellMaterial' in bpy.data.materials:
        new_shader("ShellMaterial", "glossy", 0.846874, 0.223228, 0.361307)
        bpy.data.materials["ShellMaterial"].roughness = 0.766
    if not 'BodyMaterial' in bpy.data.materials:
        new_shader("BodyMaterial", "glossy", 0.002, 0.672, 0.352)
        bpy.data.materials["BodyMaterial"].roughness = 0.766
    if not 'PlaneMaterial' in bpy.data.materials:
        new_shader("PlaneMaterial", "background", 0.0, 0.0, 0.0)


def init_collections():
    if not 'Fibers' in bpy.data.collections:
        fiber_col = bpy.data.collections.new('Fibers')
        bpy.context.scene.collection.children.link(fiber_col)
    if not 'Bodies' in bpy.data.collections:
        body_col = bpy.data.collections.new('Bodies')
        bpy.context.scene.collection.children.link(body_col)


def nurbs_cylinder(x, obj=None):
    if not obj:
        crv = bpy.data.curves.new('central_curve', 'CURVE')
        crv.dimensions = '3D'
        spline = crv.splines.new(type='NURBS')
        spline.points.add(x.shape[0] - 1)
        spline.use_endpoint_u, spline.use_endpoint_v = True, True

        for p, new_co in zip(spline.points, x):
            p.co = new_co.tolist() + [1.0]
        obj = bpy.data.objects.new('fiber', crv)
        obj.data.bevel_depth = 0.025
        obj.data.use_fill_caps = True
        obj.data.bevel_resolution = 3
        bpy.data.collections['Fibers'].objects.link(obj)
    else:
        for p, new_co in zip(obj.data.splines[0].points, x):
            p.co = new_co.tolist() + [1.0]

    # obj.hide_render = bool(x[0, 1] > 0.0)
    return obj


def new_material(label):
    mat = bpy.data.materials.get(label)
    if mat is None:
        mat = bpy.data.materials.new(name=label)

    mat.use_nodes = True
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    return mat


def new_shader(label, typename, r, g, b):
    mat = new_material(label)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')

    if typename == "diffuse":
        shader = nodes.new(type='ShaderNodeBsdfDiffuse')
        nodes["Diffuse BSDF"].inputs[0].default_value = (r, g, b, 1)
    elif typename == "emission":
        shader = nodes.new(type='ShaderNodeEmission')
        nodes["Emission"].inputs[0].default_value = (r, g, b, 1)
        nodes["Emission"].inputs[1].default_value = 1
    elif typename == "glossy":
        shader = nodes.new(type='ShaderNodeBsdfGlossy')
        nodes["Glossy BSDF"].inputs[0].default_value = (r, g, b, 1)
        nodes["Glossy BSDF"].inputs[1].default_value = 0.5
    elif typename == "principled":
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        nodes["Principled BSDF"].inputs[0].default_value = (r, g, b, 1)
        nodes["Principled BSDF"].inputs[1].default_value = 0.5
    elif typename == "background":
        shader = nodes.new(type='ShaderNodeBackground')
        nodes["Background"].inputs['Color'].default_value = (r, g, b, 1)
        nodes["Background"].inputs['Strength'].default_value = 1.0

    links.new(shader.outputs[0], output.inputs[0])

    return mat


class SkellyBlend:
    fh = None
    fpos = []
    times = []
    frame_data = None
    config_data = {}

    def __init__(self, toml_file):
        with open(toml_file, 'r') as f:
            self.config_data = toml.load(f)

        traj_file = os.path.join(os.path.dirname(toml_file), 'skelly_sim.out')

        mtime = os.stat(traj_file).st_mtime
        index_file = traj_file + '.index'
        self.fh = open(traj_file, "rb")

        if os.path.isfile(index_file):
            with open(index_file, 'rb') as f:
                print("Loading trajectory index.")
                index = pickle.load(f)
                if index['mtime'] != mtime:
                    print("Stale trajectory index file. Rebuilding.")
                    self.build_index(mtime, index_file)
                else:
                    self.fpos = index['fpos']
                    self.times = index['times']
        else:
            print("No trajectory index file. Building.")
            self.build_index(mtime, index_file)

        init_collections()
        init_materials()

        self.init_scene()

    def init_scene(self):
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = len(self) - 1
        bpy.app.handlers.frame_change_pre.append(
            lambda scene: self.draw(scene.frame_current))

        bpy.types.RenderSettings.use_lock_interface = True
        bpy.context.scene.frame_set(0)

    def __len__(self):
        return len(self.times)

    def load_frame(self, frameno):
        self.fh.seek(self.fpos[frameno])
        self.frame_data = msgpack.Unpacker(self.fh, raw=False).unpack()

    def place_periphery(self, half=True):
        if 'periphery' not in self.config_data:
            print("No periphery found in configuration data.")
            return

        periphery = self.config_data['periphery']
        if periphery['shape'] == 'sphere':
            place_shell(self.config_data['periphery']['radius'], half)
        else:
            print("Periphery of type '{}' not yet supported".format(
                periphery['shape']))

    def draw(self, frameno):
        self.load_frame(frameno)

        fibdata = self.frame_data['fibers'][0]
        init = len(bpy.data.collections['Fibers'].objects) == 0

        for ifib in range(0, len(fibdata)):
            pos = np.array(fibdata[ifib]['x_'][3:])
            pos = pos.reshape(pos.size // 3, 3)

            if init:
                newfib = nurbs_cylinder(pos)
                fibmat = bpy.data.materials['FiberMaterial']
                newfib.data.materials.append(fibmat)
            else:
                nurbs_cylinder(pos,
                               bpy.data.collections['Fibers'].objects[ifib])


        bodydata = self.frame_data['bodies'][0]
        init = len(bpy.data.collections['Bodies'].objects) == 0

        for ibody in range(0, len(bodydata)):
            pos = bodydata[ibody]['position_'][3:]

            if init:
                create_body(pos, bodydata[ibody]['radius_'])
            else:
                body = bpy.data.collections['Bodies'].objects[ibody]
                body.location = pos


    def build_index(self, mtime, index_file):
        unpacker = msgpack.Unpacker(self.fh, raw=False)

        while True:
            try:
                self.fpos.append(unpacker.tell())
                n_keys = unpacker.read_map_header()
                for key in range(n_keys):
                    key = unpacker.unpack()
                    if key == 'time':
                        self.times.append(unpacker.unpack())
                    else:
                        unpacker.skip()

            except msgpack.exceptions.OutOfData:
                self.fpos.pop()
                break

        index = {
            'mtime': mtime,
            'fpos': self.fpos,
            'times': self.times,
        }
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)


if __name__ == "__main__":
    if '--' not in sys.argv:
        print(
            "skelly_blend error! Must supply trailing `-- myconfig.toml` to command line"
        )
        sys.exit()

    argv = sys.argv[sys.argv.index("--") + 1:]

    init(argv[0])
