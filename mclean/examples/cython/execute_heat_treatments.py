from timeit import timeit

# Benchmark the get_T_and_D function
t1 = timeit("get_T_and_D(t_tot=1000, Tstart=1000, D0=0.5, dQ=1.9e-9, Tend=100, dt=0.1, heat_treatment_type='linear', r=-1)",
            setup="from heat_treatment import get_T_and_D",
            number=10000)

# Benchmark the get_T_and_D function
t2 = timeit("get_T_and_D(t_tot=1000, Tstart=1000, D0=0.5, dQ=1.9e-9, Tend=100.0, dt=0.1, heat_treatment_type='linear', r=-1.0)",
            setup="from heat_treatment_annotated import get_T_and_D",
            number=10000)

t3 = timeit("get_T_and_D(1000, 1000.0, 0.5, 1e-9, 1.0, 100.0, 'linear', -1.0)",
            setup="from heat_treatment_cythonized import get_T_and_D",
            number=10000)

print(f"Python: Time for get_T_and_D: {t1} seconds for 10k runs")
print(f"Annotated: Time for get_T_and_D is {t1/t2} times faster than python for 10k runs")
print(f"Cythonized: Time for get_T_and_D is {t1/t3} times faster than python for 10k runs")
