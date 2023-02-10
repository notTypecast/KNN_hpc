import matplotlib.pyplot as plt

xp = [65536, 131072, 262144, 524288, 1048576, 2097152]
y_org = [2868.592262, 5730.572939, 11445.715904, 22863.208294, 45940.021992, 91917.198420]
y_seq = [831.532717, 1649.008036, 3316.325665, 6556.166887, 13018.276215, 26091.174364]
y_omp = [406.803846, 852.778673, 1815.611839, 3344.151974, 6379.420996, 12576.691628]
y_omp_queries = [275.177002, 569.693089, 1294.790030, 2557.402849, 5691.832066, 11926.064014]
y_mpi = [303.951979, 741.048336, 1607.579231, 3033.519983, 6103.737354, 12027.818680]
y_cuda_1060 = [74.177980, 121.797085, 182.941198, 369.824886, 997.249126, 5911.599398]
y_cuda_950 = [105.492115, 162.564754, 302.945852, 745.904207, 2382.526875, 7759.004593]
y_cuda_tesla = [120.176077, 189.937830, 318.902969, 566.091061, 1053.445101, 3411.099195]
y_openacc_tesla = [127.877951, 195.807934, 322.844982, 566.100836, 1044.625044, 3343.361139]

plt.plot(xp, list(x/1000 for x in y_org), label="Original code")
plt.plot(xp, list(x/1000 for x in y_seq), label="Improved sequential code")
plt.plot(xp, list(x/1000 for x in y_omp), label="OpenMP parallelized code")
plt.plot(xp, list(x/1000 for x in y_omp_queries), label="OpenMP query parallelized code")
plt.plot(xp, list(x/1000 for x in y_mpi), label="MPI parallelized code")
plt.plot(xp, list(x/1000 for x in y_cuda_950), label="CUDA code on GTX 950 2GB")
plt.plot(xp, list(x/1000 for x in y_cuda_1060), label="CUDA code on GTX 1060 3GB")
plt.plot(xp, list(x/1000 for x in y_cuda_tesla), label="CUDA code on Tesla K40c")
plt.plot(xp, list(x/1000 for x in y_openacc_tesla), label="OpenACC code on Tesla K40c")
plt.yscale("log")
plt.legend()
plt.xlabel("Total training points")
plt.ylabel("Execution time (s)")
plt.tight_layout(pad=0)
figure = plt.gcf()
figure.set_size_inches(12, 10)
plt.savefig("../all_plots.png", dpi=100)
#plt.show()
