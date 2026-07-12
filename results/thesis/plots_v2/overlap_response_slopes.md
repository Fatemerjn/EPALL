# Pooled measured-overlap regressions

WorstDrop is kept signed: negative values are repair/improvement, not clipped damage.
The 95% intervals use HC3 heteroskedasticity-robust OLS standard errors with
Student-t critical values. Exact
config/seed retries are de-duplicated to their latest completed run. The x measure is
PALL-Adapter shared-critical ratio, but full-network PALL mean subnet-mask IoU; slope
magnitudes across those representations therefore need cautious comparison.

| Dataset | Role | Method | x measure | n | unique x | x range | Slope | 95% CI | CI includes 0 | R2 | Negative WorstDrop |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| cifar100 | primary | pall_original | mean subnet-mask IoU | 9 | 3 | 0.0526--0.1110 | 0.565207 | [-411.195708, 412.326122] | True | 0.4549 | 0 |
| cifar100 | primary | pall_modified | mean subnet-mask IoU | 10 | 4 | 0.0526--0.1110 | 0.102824 | [-0.234162, 0.439809] | True | 0.3399 | 0 |
| cifar100 | primary | pall_adapter | shared-critical ratio | 39 | 34 | 0.3743--0.9567 | 0.019706 | [-0.034594, 0.074006] | True | 0.0113 | 0 |
| cifar10 | secondary | pall_original | mean subnet-mask IoU | 11 | 4 | 0.1110--0.1112 | -53.749386 | [-233.218529, 125.719756] | True | 0.0653 | 0 |
| cifar10 | secondary | pall_modified | mean subnet-mask IoU | 12 | 5 | 0.1110--0.1112 | -28.469080 | [-182.415747, 125.477587] | True | 0.0237 | 0 |
| cifar10 | secondary | pall_adapter | shared-critical ratio | 49 | 20 | 0.4107--0.7882 | -0.012147 | [-0.049112, 0.024819] | True | 0.0081 | 16 |
