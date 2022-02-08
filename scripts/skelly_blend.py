import sys
import site
import math
import bpy
import numpy as np
import bmesh
import mathutils
import time
import os

if '--' not in sys.argv:
    print(
        "skelly_blend error! Must supply trailing `-- myconfig.toml` to command line"
    )
    sys.exit()

argv = sys.argv[sys.argv.index("--") + 1:]

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


def nurbs_cylinder(x, obj=None):
    if not obj:
        crv = bpy.data.curves.new('central_curve', 'CURVE')
        crv.dimensions = '3D'
        spline = crv.splines.new(type='NURBS')
        spline.points.add(x.shape[0] - 1)
        spline.use_endpoint_u, spline.use_endpoint_v = True, True

        for p, new_co in zip(spline.points, x):
            p.co = new_co.tolist() + [1.0]
        obj = bpy.data.objects.new('cylinder', crv)
        obj.data.bevel_depth = 0.025
        obj.data.use_fill_caps = True
        obj.data.bevel_resolution = 3
        bpy.data.collections['Fibers'].objects.link(obj)
    else:
        for p, new_co in zip(obj.data.splines[0].points, x):
            p.co = new_co.tolist() + [1.0]

    obj.hide_render = bool(x[0, 1] > 0.0)
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
        nodes["Background"].inputs['Strength'].default_value = 10.0

    links.new(shader.outputs[0], output.inputs[0])

    return mat


class SkellySim:
    fh = None
    fpos = []
    times = []
    frame_data = None

    def __init__(self, toml_file):
        with open(toml_file, 'r') as f:
            td = toml.load(f)
        if 'periphery' in td:
            periphery = td['periphery']
            if periphery['shape'] == 'sphere':
                place_half_shell(td['periphery']['radius'])
            else:
                print("Periphery of type '{}' not yet supported".format(
                    periphery['shape']))

        traj_file = os.path.join(os.path.dirname(toml_file), 'skelly_sim.out')

        self.fh = open(traj_file, "rb")
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

    def __len__(self):
        return len(self.times)

    def load_frame(self, frameno):
        self.fh.seek(self.fpos[frameno])
        self.frame_data = msgpack.Unpacker(self.fh, raw=False).unpack()

    def draw(self, frameno):
        self.load_frame(frameno)

        data = self.frame_data['fibers'][0]
        init = len(bpy.data.collections['Fibers'].objects) == 0

        for ifib in range(0, len(data)):
            pos = np.array(data[ifib]['x_'][3:])
            pos = pos.reshape(pos.size // 3, 3)

            if init:
                newfib = nurbs_cylinder(pos)
                fibmat = bpy.data.materials['FiberMaterial']
                newfib.data.materials.append(fibmat)
            else:
                nurbs_cylinder(pos,
                               bpy.data.collections['Fibers'].objects[ifib])


def clear_scene():
    if "Cube" in bpy.data.meshes:
        cube = bpy.data.objects["Cube"]
        bpy.data.objects.remove(cube, do_unlink=True)
    if "Light" in bpy.data.objects:
        light = bpy.data.objects["Light"]
        bpy.data.objects.remove(light, do_unlink=True)


def place_half_shell(radius):
    mesh = bpy.data.meshes.new('Sphere')
    sphere = bpy.data.objects.new("Sphere", mesh)
    sphere.data.materials.append(bpy.data.materials['ShellMaterial'])
    bpy.context.collection.objects.link(sphere)

    bpy.context.view_layer.objects.active = sphere
    sphere.select_set(True)
    modifier = sphere.modifiers.new(name="Solidify", type="SOLIDIFY")
    modifier.thickness = 0.1
    modifier.offset = 1.0

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm,
                              u_segments=64,
                              v_segments=32,
                              diameter=radius)

    for vert in bm.verts:
        if vert.co[1] > 0:
            bm.verts.remove(vert)
    bm.to_mesh(mesh)
    bpy.ops.object.shade_smooth()
    bm.free()


def place_backdrop():
    bpy.ops.mesh.primitive_plane_add(size=100.0,
                                     location=(0.0, -5.5, 0.0),
                                     rotation=(0.5 * np.pi, 0.0, 0.0))
    plane = bpy.data.objects['Plane']
    plane.data.materials.append(bpy.data.materials['PlaneMaterial'])


def place_camera():
    if "Camera" not in bpy.data.cameras:
        camera_data = bpy.data.cameras.new(name='Camera')
        camera_object = bpy.data.objects.new('Camera', camera_data)
        bpy.context.scene.collection.objects.link(camera_object)

    camera = bpy.data.objects['Camera']
    mat_loc = mathutils.Matrix.Translation((7.1, 21.2, -10.9))
    eul = mathutils.Euler(
        (math.radians(-115), math.radians(1), math.radians(-18.4)), 'XYZ')
    camera.matrix_world = mat_loc @ eul.to_matrix().to_4x4()


def set_light():
    light_data = bpy.data.lights.new(name="Light", type='SUN')
    light_data.energy = 2.0
    light_data.specular_factor = 1.0
    light = bpy.data.objects.new(name="Light", object_data=light_data)
    bpy.context.collection.objects.link(light)

    mat_loc = mathutils.Matrix.Translation((0.0, 30, 0.0))
    eul = mathutils.Euler(
        (math.radians(90.0), math.radians(0), math.radians(180.0)), 'XYZ')
    light.matrix_world = mat_loc @ eul.to_matrix().to_4x4()


def set_ambient_light():
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[
        'Color'].default_value = (1.0, 1.0, 1.0, 1)
    bpy.context.scene.world.node_tree.nodes['Background'].inputs[
        'Strength'].default_value = 0.7


def init_materials():
    new_shader("FiberMaterial", "glossy", 0.087, 0.381, 1.0)
    new_shader("ShellMaterial", "principled", 0.233, 0.233, 0.233)
    new_shader("PlaneMaterial", "background", 1.0, 1.0, 1.0)


def init_collections():
    fiber_col = bpy.data.collections.new('Fibers')
    bpy.context.scene.collection.children.link(fiber_col)


def init_scene(sim: SkellySim):
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(sim) - 1
    bpy.app.handlers.frame_change_pre.append(
        lambda scene: sim.draw(scene.frame_current))

    bpy.types.RenderSettings.use_lock_interface = True
    bpy.context.scene.frame_set(0)


def main():
    init_collections()
    init_materials()
    clear_scene()
    place_backdrop()
    place_camera()
    set_light()
    set_ambient_light()

    sim = SkellySim(argv[0])
    init_scene(sim)

    if len(argv) > 1:
        render_dir = argv[1]
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'PNG'
        for i in range(0, len(sim)):
            scene.frame_set(i)
            scene.render.filepath = os.path.join(
                render_dir, "frame_{}.png".format(str(i).rjust(5, '0')))
            bpy.ops.render.render(write_still=1)


if __name__ == "__main__":
    main()
