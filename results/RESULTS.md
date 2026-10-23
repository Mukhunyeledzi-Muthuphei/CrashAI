# Test results

250 emergencies per scenario type (1000 per controller), identical for every controller.

Each cell: **crash rate** (mean impact speed over all episodes, 0 when avoided).

## By scenario

| Controller | Overall | stopped | lead_brake | cut_in | pedestrian |
|---|---|---|---|---|---|
| driver_only | 87% (46 km/h) | 100% (78 km/h) | 100% (53 km/h) | 88% (30 km/h) | 59% (24 km/h) |
| brake_no_abs | 49% (17 km/h) | 59% (24 km/h) | 93% (30 km/h) | 41% (13 km/h) | 4% (1 km/h) |
| brake_abs | 44% (14 km/h) | 48% (18 km/h) | 87% (24 km/h) | 40% (13 km/h) | 2% (0 km/h) |
| brake_swerve | 30% (11 km/h) | 30% (13 km/h) | 41% (15 km/h) | 42% (15 km/h) | 9% (1 km/h) |
| brake_swerve_esc | 29% (11 km/h) | 28% (13 km/h) | 40% (15 km/h) | 39% (14 km/h) | 9% (1 km/h) |
| learned_pedal | 13% (4 km/h) | 8% (4 km/h) | 18% (4 km/h) | 25% (8 km/h) | 2% (0 km/h) |
| learned_full | 15% (5 km/h) | 9% (4 km/h) | 24% (6 km/h) | 26% (9 km/h) | 1% (0 km/h) |
| learned_brakes_only | 25% (7 km/h) | 17% (6 km/h) | 48% (12 km/h) | 33% (9 km/h) | 2% (0 km/h) |


## By road surface

| Controller | surface:dry | surface:wet | surface:snow | surface:ice |
|---|---|---|---|---|
| driver_only | 85% (51 km/h) | 87% (49 km/h) | 90% (36 km/h) | 91% (29 km/h) |
| brake_no_abs | 38% (14 km/h) | 51% (19 km/h) | 61% (19 km/h) | 83% (20 km/h) |
| brake_abs | 33% (11 km/h) | 45% (16 km/h) | 58% (17 km/h) | 79% (19 km/h) |
| brake_swerve | 17% (7 km/h) | 27% (11 km/h) | 49% (17 km/h) | 78% (22 km/h) |
| brake_swerve_esc | 16% (7 km/h) | 25% (10 km/h) | 48% (17 km/h) | 78% (22 km/h) |
| learned_pedal | 5% (2 km/h) | 9% (3 km/h) | 21% (8 km/h) | 54% (16 km/h) |
| learned_full | 8% (3 km/h) | 9% (3 km/h) | 22% (7 km/h) | 58% (18 km/h) |
| learned_brakes_only | 13% (4 km/h) | 19% (5 km/h) | 40% (11 km/h) | 80% (20 km/h) |

