# Pooled measured-overlap regressions

WorstDrop is kept signed: negative values are repair/improvement, not clipped damage.
The 95% intervals use HC3 heteroskedasticity-robust OLS standard errors with
Student-t critical values. Exact
config/seed retries are de-duplicated to their latest completed run. The x measure is
PALL-Adapter shared-critical ratio, but full-network PALL mean subnet-mask IoU; slope
magnitudes across those representations therefore need cautious comparison.

| Dataset | Role | Method | x measure | n | unique x | x range | Slope | 95% CI | CI includes 0 | R2 | Negative WorstDrop |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| cifar100 | primary | pall_original | mean subnet-mask IoU | 11 | 3 | 0.0526--0.1110 | 0.517337 | [-1124.372381, 1125.407056] | True | 0.3574 | 0 |
| cifar100 | primary | pall_modified | mean subnet-mask IoU | 12 | 4 | 0.0526--0.1110 | 0.065167 | [-0.267707, 0.398040] | True | 0.0564 | 0 |
| cifar100 | primary | pall_adapter | shared-critical ratio | 86 | 57 | 0.0000--0.9567 | 0.038775 | [0.022733, 0.054817] | False | 0.1288 | 0 |
| cifar10 | secondary | pall_original | mean subnet-mask IoU | 12 | 4 | 0.1110--0.1112 | -35.945082 | [-209.282454, 137.392289] | True | 0.0316 | 0 |
| cifar10 | secondary | pall_modified | mean subnet-mask IoU | 14 | 5 | 0.1110--0.1112 | 3.863961 | [-107.690736, 115.418658] | True | 0.0005 | 0 |
| cifar10 | secondary | pall_adapter | shared-critical ratio | 92 | 39 | 0.0000--0.7882 | 0.008471 | [0.000298, 0.016645] | False | 0.0271 | 18 |
