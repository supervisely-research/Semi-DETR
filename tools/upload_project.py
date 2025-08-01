import supervisely as sly

api = sly.Api()

p = sly.Project("/root/volume/kanal75/run_1_iter_50000", sly.OpenMode.READ)

sly.Project.upload(
    dir="/root/volume/kanal75/run_1_iter_50000",
    api=api,
    workspace_id=80,
    project_name="Semi-DETR run_1 iter_50000",
)