# DFLASH Sweep: gsm8k, mt-bench

## Settings
- data_names: `gsm8k, mt-bench`
- target_model: `Qwen/Qwen3.6-27B`
- draft_model: `z-lab/Qwen3.6-27B-DFlash`
- prompt_style: `chat`
- max_new_tokens: `1024`
- attention_backends: `triton`
- tp_sizes: `1`
- concurrencies: `1, 8, 32`
- samples_per_concurrency: `base=8`
- device_sm: `100`
- is_blackwell: `True`
- skip_baseline: `True`

Note: This sweep focuses on throughput. Correctness is not evaluated for this generic dataset script.

# Results for Dataset: `gsm8k`
## Backend: `triton`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 353.08 | 820.48 | 1,098.38 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 8.0 | 30.7 | 88.3 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2827 | 25205 | 97002 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 8.481 | 7.756 | 7.974 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 8.0 | 30.7 | 88.3 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2827 | 25205 | 97002 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 8.481 | 7.756 | 7.974 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 339 | 3211 | 11939 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

# Results for Dataset: `mt-bench`
## Backend: `triton`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 155.43 | 785.23 | 709.93 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 11.4 | 34.3 | 175.2 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1771 | 26970 | 124390 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 3.154 | 5.478 | 5.003 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 11.4 | 34.3 | 175.2 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1771 | 26970 | 124390 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 3.154 | 5.478 | 5.003 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 552 | 4701 | 26141 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

