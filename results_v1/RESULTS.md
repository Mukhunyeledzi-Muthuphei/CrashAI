# Test results

250 emergencies per scenario type (1000 per controller), identical for every controller.

Each cell: **crash rate** (mean impact speed over all episodes, 0 when avoided).

## By scenario

| Controller | Overall | stopped | lead_brake | cut_in | pedestrian |
|---|---|---|---|---|---|
| driver_only | 88% (47 km/h) | 100% (78 km/h) | 100% (53 km/h) | 95% (34 km/h) | 59% (24 km/h) |
| brake_no_abs | 57% (19 km/h) | 59% (24 km/h) | 93% (30 km/h) | 73% (20 km/h) | 4% (1 km/h) |
| brake_abs | 52% (16 km/h) | 48% (18 km/h) | 87% (24 km/h) | 72% (20 km/h) | 2% (0 km/h) |
| brake_swerve | 38% (13 km/h) | 32% (15 km/h) | 43% (16 km/h) | 70% (19 km/h) | 9% (1 km/h) |
| brake_swerve_esc | 38% (12 km/h) | 30% (14 km/h) | 42% (15 km/h) | 69% (19 km/h) | 9% (1 km/h) |
| learned_pedal | 44% (16 km/h) | 59% (25 km/h) | 48% (19 km/h) | 66% (19 km/h) | 1% (0 km/h) |
| learned_full | 44% (17 km/h) | 66% (32 km/h) | 43% (16 km/h) | 68% (20 km/h) | 0% (0 km/h) |
| learned_brakes_only | 64% (26 km/h) | 93% (47 km/h) | 87% (36 km/h) | 74% (21 km/h) | 2% (0 km/h) |


## By road surface

| Controller | surface:dry | surface:wet | surface:snow | surface:ice |
|---|---|---|---|---|
| driver_only | 88% (52 km/h) | 88% (49 km/h) | 91% (37 km/h) | 91% (29 km/h) |
| brake_no_abs | 50% (17 km/h) | 56% (20 km/h) | 69% (21 km/h) | 83% (21 km/h) |
| brake_abs | 44% (13 km/h) | 50% (17 km/h) | 68% (20 km/h) | 79% (20 km/h) |
| brake_swerve | 28% (9 km/h) | 32% (12 km/h) | 55% (18 km/h) | 86% (24 km/h) |
| brake_swerve_esc | 27% (9 km/h) | 32% (12 km/h) | 54% (18 km/h) | 86% (24 km/h) |
| learned_pedal | 43% (16 km/h) | 39% (14 km/h) | 40% (15 km/h) | 64% (21 km/h) |
| learned_full | 41% (17 km/h) | 39% (16 km/h) | 51% (18 km/h) | 64% (20 km/h) |
| learned_brakes_only | 61% (28 km/h) | 63% (28 km/h) | 65% (21 km/h) | 80% (21 km/h) |

