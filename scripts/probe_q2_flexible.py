import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from dproblem.config import load_model_config
from dproblem.domain.geometry import RasterDEM
from dproblem.io.dataset import load_project_data
from dproblem.q2.route import RouteEvaluator
from dproblem.q2.scheduler import q1_batches_as_flexible_routes, schedule_routes

data = load_project_data(project_root)
config = load_model_config(project_root)
evaluator = RouteEvaluator(
    RasterDEM(data["paths"]["dem_tif"]),
    data["nodes"]["centers"][0],
    data["nodes"]["service_areas"],
    config["energy_model"],
)
routes = q1_batches_as_flexible_routes(project_root, data["demands"]["boxes"])
# Urgency repair for services split over multiple Q1 batches: move hard boxes
# into the earliest batch and exchange like-for-like soft boxes when needed.
s001 = [route for route in routes if route["visit_order"] == ["S001"]]
early = next(route for route in s001 if any(box["货箱编号"] == "S001-MED-01" for box in route["boxes"]))
late = next(route for route in s001 if any(box["货箱编号"] == "S001-WAT-01" for box in route["boxes"]))
hard_water = next(box for box in late["boxes"] if box["货箱编号"] == "S001-WAT-01")
soft_water = next(box for box in early["boxes"] if box["货箱编号"] == "S001-WAT-06")
early["boxes"].remove(soft_water)
late["boxes"].remove(hard_water)
early["boxes"].append(hard_water)
late["boxes"].append(soft_water)
result = schedule_routes(routes, data, config, evaluator)
print(json.dumps({key: value for key, value in result.items() if key not in ("trips", "battery_rows", "deliveries")}, ensure_ascii=False, indent=2))
if "trips" in result:
    print(json.dumps([
        {
            "trip": trip["架次编号"], "type": trip["机型编号"], "drone": trip["无人机编号"],
            "start": trip["开始时刻（s）"], "route": trip["访问服务区顺序"],
            "return": trip["返回O01时刻（s）"]
        }
        for trip in result["trips"]
    ], ensure_ascii=False, indent=2))
