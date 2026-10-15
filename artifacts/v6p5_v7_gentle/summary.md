# v6p5-23 gentle fine-tune — 2026-10-15 20:34

## Teachers, MEAN_5 two-tier @0.05 (val_clean_v6)
| model      |   MEAN_5 |
|:-----------|---------:|
| v5_orig    |   0.8283 |
| soup_g_v6  |   0.8281 |
| soup_gf_v6 |   0.8266 |
| soup_ft_v6 |   0.8255 |
| gf_v6      |   0.8252 |
| soup_g_v5  |   0.8242 |
| g_v6       |   0.8219 |
| g_v5       |   0.8206 |
| ft_v5      |   0.8192 |
| ft_v6      |   0.8129 |

## Teacher paired bootstrap
| split        | compare               |    diff |   ci95_lo |   ci95_hi |   P(>0) |
|:-------------|:----------------------|--------:|----------:|----------:|--------:|
| val_clean_v6 | ft_v5 − v5_orig       | -0.0089 |   -0.0198 |    0.0016 |  0.0525 |
| val_clean_v6 | ft_v6 − v5_orig       | -0.0143 |   -0.028  |   -0.0003 |  0.0245 |
| val_clean_v6 | g_v6 − v5_orig        | -0.0062 |   -0.014  |    0.0001 |  0.0275 |
| val_clean_v6 | g_v5 − v5_orig        | -0.0069 |   -0.0126 |   -0.0017 |  0.0045 |
| val_clean_v6 | gf_v6 − v5_orig       | -0.0029 |   -0.0102 |    0.0053 |  0.2235 |
| val_clean_v6 | soup_ft_v6 − v5_orig  | -0.0027 |   -0.0116 |    0.0065 |  0.274  |
| val_clean_v6 | soup_g_v6 − v5_orig   |  0      |   -0.0027 |    0.0028 |  0.51   |
| val_clean_v6 | soup_gf_v6 − v5_orig  | -0.0015 |   -0.0049 |    0.0025 |  0.1905 |
| val_clean_v6 | soup_g_v5 − v5_orig   | -0.0039 |   -0.0085 |   -0.0002 |  0.018  |
| val_clean_v6 | g_v6 − g_v5           |  0.0007 |   -0.0056 |    0.0071 |  0.598  |
| val_clean_v6 | soup_g_v6 − soup_g_v5 |  0.0039 |   -0.0004 |    0.009  |  0.9565 |
| val_clean    | ft_v5 − v5_orig       | -0.0077 |   -0.0189 |    0.0028 |  0.081  |
| val_clean    | ft_v6 − v5_orig       | -0.014  |   -0.0276 |    0.0001 |  0.026  |
| val_clean    | g_v6 − v5_orig        | -0.0061 |   -0.0137 |    0.0004 |  0.035  |
| val_clean    | g_v5 − v5_orig        | -0.0067 |   -0.0125 |   -0.0017 |  0.006  |
| val_clean    | gf_v6 − v5_orig       | -0.0032 |   -0.0109 |    0.0056 |  0.2095 |
| val_clean    | soup_ft_v6 − v5_orig  | -0.0023 |   -0.0112 |    0.0071 |  0.2975 |
| val_clean    | soup_g_v6 − v5_orig   |  0      |   -0.0027 |    0.003  |  0.504  |
| val_clean    | soup_gf_v6 − v5_orig  | -0.0013 |   -0.005  |    0.0032 |  0.237  |
| val_clean    | soup_g_v5 − v5_orig   | -0.004  |   -0.0088 |   -0.0003 |  0.017  |
| val_clean    | g_v6 − g_v5           |  0.0006 |   -0.006  |    0.0072 |  0.5805 |
| val_clean    | soup_g_v6 − soup_g_v5 |  0.0041 |   -0.0004 |    0.0095 |  0.959  |

Best teacher: **v5_orig** (student distilled from it)

## Students, Pi-only MEAN_5
| split        | model      |   Pi-only MEAN_5 |   insulator_broken |   insulator_flashover |   bird_nest |   foreign_object |   broken_strand |
|:-------------|:-----------|-----------------:|-------------------:|----------------------:|------------:|-----------------:|----------------:|
| val_clean_v6 | n_v5       |           0.7785 |             0.8128 |                0.7731 |      0.908  |           0.6264 |          0.7721 |
| val_clean    | n_v5       |           0.7792 |             0.8128 |                0.7731 |      0.908  |           0.6298 |          0.7721 |
| val_clean_v6 | n_v6       |           0.7697 |             0.8099 |                0.7795 |      0.9048 |           0.6038 |          0.7504 |
| val_clean    | n_v6       |           0.7696 |             0.8099 |                0.7795 |      0.9048 |           0.6034 |          0.7504 |
| val_clean_v6 | gn_v6      |           0.7776 |             0.7952 |                0.7662 |      0.9096 |           0.6373 |          0.7797 |
| val_clean    | gn_v6      |           0.7764 |             0.7952 |                0.7662 |      0.9096 |           0.6314 |          0.7797 |
| val_clean_v6 | soup_n_v6  |           0.7721 |             0.8114 |                0.7731 |      0.9086 |           0.6114 |          0.7561 |
| val_clean    | soup_n_v6  |           0.7726 |             0.8114 |                0.7731 |      0.9086 |           0.614  |          0.7561 |
| val_clean_v6 | soup_gn_v6 |           0.7779 |             0.8111 |                0.7726 |      0.9068 |           0.6248 |          0.7743 |
| val_clean    | soup_gn_v6 |           0.7777 |             0.8111 |                0.7726 |      0.9068 |           0.6238 |          0.7743 |

## Student paired bootstrap
| split        | compare           |    diff |   ci95_lo |   ci95_hi |   P(>0) |
|:-------------|:------------------|--------:|----------:|----------:|--------:|
| val_clean_v6 | n_v6 − n_v5       | -0.0083 |   -0.018  |    0.001  |  0.04   |
| val_clean_v6 | gn_v6 − n_v5      | -0.0012 |   -0.0127 |    0.0098 |  0.404  |
| val_clean_v6 | soup_n_v6 − n_v5  | -0.0059 |   -0.0126 |    0.0004 |  0.0355 |
| val_clean_v6 | soup_gn_v6 − n_v5 | -0.0008 |   -0.0108 |    0.0098 |  0.428  |
| val_clean    | n_v6 − n_v5       | -0.0091 |   -0.0188 |    0.0002 |  0.0295 |
| val_clean    | gn_v6 − n_v5      | -0.0029 |   -0.0141 |    0.008  |  0.298  |
| val_clean    | soup_n_v6 − n_v5  | -0.0059 |   -0.0125 |    0.0005 |  0.037  |
| val_clean    | soup_gn_v6 − n_v5 | -0.0014 |   -0.0113 |    0.0094 |  0.3795 |