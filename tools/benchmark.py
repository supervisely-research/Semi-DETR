from supervisely.nn.benchmark.object_detection.benchmark import ObjectDetectionBenchmark
import supervisely as sly

api = sly.Api()

benchmark = ObjectDetectionBenchmark(
    api,
    gt_project_id=2847,
    gt_dataset_ids=[14970],
    output_dir='output',
)

benchmark.evaluate(2895)
# benchmark.visualize()

bm = benchmark
eval_res_dir = "/model-benchmark/Semi-DETR run_1 iter_50000"
bm.upload_eval_results(eval_res_dir + "/evaluation/")

# 7. Prepare visualizations, report and upload
bm.visualize()
_ = bm.upload_visualizations(eval_res_dir + "/visualizations/")
lnk_file_info = bm.lnk
report = bm.report
report_id = bm.report.id
eval_metrics = bm.key_metrics
primary_metric_name = bm.primary_metric_name
bm.upload_report_link(eval_res_dir, report_id, None)
