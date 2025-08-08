bl_info = {
    "name": "Hunyuan3D Multiview Generator",
    "author": "jino1113",
    "version": (1, 2),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Hunyuan3D",
    "description": "Generate 3D model from 4-view images using Hunyuan AI API",
    "category": "3D View",
}

import bpy
import os
import base64
import requests
import tempfile

# ---------- Encode Function ----------
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Failed to encode {image_path}: {e}")
        return ""

# ---------- Properties ----------
class MultiviewProperties(bpy.types.PropertyGroup):
    api_url: bpy.props.StringProperty(
        name="API URL",
        description="URL to the backend API",
        default="http://127.0.0.1:8080/multiview"
    )
    front_path: bpy.props.StringProperty(name="Front View", subtype='FILE_PATH')
    back_path: bpy.props.StringProperty(name="Back View", subtype='FILE_PATH')
    left_path: bpy.props.StringProperty(name="Left View", subtype='FILE_PATH')
    right_path: bpy.props.StringProperty(name="Right View", subtype='FILE_PATH')

    octree_resolution: bpy.props.IntProperty(name="Octree Resolution", default=512)
    num_inference_steps: bpy.props.IntProperty(name="Number of Inference Steps", default=20)
    guidance_scale: bpy.props.FloatProperty(name="Guidance Scale", default=5.5)
    generate_texture: bpy.props.BoolProperty(name="Generate Texture", default=True)

# ---------- Panel UI ----------
class VIEW3D_PT_MultiviewPanel(bpy.types.Panel):
    bl_label = "Multiview Prompt"
    bl_idname = "VIEW3D_PT_multiview_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hunyuan3D"

    def draw(self, context):
        layout = self.layout
        mv_props = context.scene.mv_props

        layout.label(text="API Configuration")
        layout.prop(mv_props, "api_url")

        layout.label(text="Upload 4-view images")
        layout.prop(mv_props, "front_path")
        layout.prop(mv_props, "back_path")
        layout.prop(mv_props, "left_path")
        layout.prop(mv_props, "right_path")

        layout.prop(mv_props, "octree_resolution")
        layout.prop(mv_props, "num_inference_steps")
        layout.prop(mv_props, "guidance_scale")
        layout.prop(mv_props, "generate_texture")

        layout.operator("object.generate_from_multiview", text="Generate from Multiview")

# ---------- Operator ----------
class OBJECT_OT_GenerateFromMultiview(bpy.types.Operator):
    bl_idname = "object.generate_from_multiview"
    bl_label = "Generate from Multiview"

    def execute(self, context):
        mv = context.scene.mv_props
        views = {
            "front": mv.front_path,
            "back": mv.back_path,
            "left": mv.left_path,
            "right": mv.right_path,
        }

        encoded_views = {}

        # ตรวจแค่มุมที่ผู้ใช้ใส่
        for name, path in views.items():
            if path.strip() == "":
                continue  # ข้ามถ้าไม่ได้ใส่ภาพ

            abspath = bpy.path.abspath(path)
            if not os.path.exists(abspath):
                self.report({'ERROR'}, f"File not found for {name} view: {path}")
                return {'CANCELLED'}

            b64 = encode_image_to_base64(abspath)
            encoded_views[name] = b64

        # Include parameters
        payload = {
            **encoded_views,
            "octree_resolution": mv.octree_resolution,
            "num_inference_steps": mv.num_inference_steps,
            "guidance_scale": mv.guidance_scale,
            "texture": mv.generate_texture
        }

        # Call backend API
        try:
            response = requests.post(mv.api_url, json=payload)

            if response.status_code == 200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
                temp_file.write(response.content)
                temp_file.close()
                bpy.ops.import_scene.gltf(filepath=temp_file.name)
                self.report({'INFO'}, "Model imported successfully.")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"API error {response.status_code}: {response.text}")
        except Exception as e:
            self.report({'ERROR'}, f"API call failed: {e}")
            return {'CANCELLED'}

        return {'CANCELLED'}

# ---------- Register ----------
classes = (
    MultiviewProperties,
    VIEW3D_PT_MultiviewPanel,
    OBJECT_OT_GenerateFromMultiview,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mv_props = bpy.props.PointerProperty(type=MultiviewProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mv_props

if __name__ == "__main__":
    register()
