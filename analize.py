import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np

from pathlib import Path
dir = "runs/run8/" 
dir_anal = dir + "analyze/"
Path(dir_anal).mkdir(exist_ok=True)

df = pd.read_csv(dir + "metrics.csv")

# generation,generator_loss,gen_non_penalty_rate,ga_non_penalty_rate,best_fit,mean_fit,average_length,lr

ground_truth = -14.644 # -677.509 # 
plt.plot(df["generation"]+1, df["generator_loss"], color="blue", label="Generator Loss")
plt.axhline(y=0, color='r', linestyle='-')
plt.title("Generator Loss")
plt.xlabel("Generations")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(dir_anal + "generator_loss.png")
plt.clf()

fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

fig.suptitle("GA genomes metrics summary")

axs[0, 0].plot(df["generation"] + 1, df["best_energy"], color="cyan", label="Best fit")
axs[0, 0].axhline(y=ground_truth, color="r", linestyle="--", label=fr"Ground Truth $\approx$ {ground_truth}")
axs[0, 0].set_title("Best genome fit value of each generation")
axs[0, 0].set_ylabel("units")
axs[0, 0].legend()

axs[0, 1].plot(df["generation"]+1, np.abs(np.array(ground_truth - df["best_energy"])), label=r"$err = | y_{true} - y_{pred} |$", color="orange")
axs[0, 1].set_title("Best genome error from ground truth")
axs[0, 1].set_xlabel("Generations")
axs[0, 1].set_ylabel("units")
axs[0, 1].legend()

axs[1, 0].plot(df["generation"] + 1, df["average_length"], color="blue", label="Average length")
axs[1, 0].set_title("Average length of alpha array")
axs[1, 0].set_xlabel("Generations")
axs[1, 0].set_ylabel("count")
axs[1, 0].legend()

axs[1, 1].plot(df["generation"]+1, df["gen_non_penalty_rate"], label="Generator non penalty rate", color="red")
axs[1, 1].plot(df["generation"]+1, df["ga_non_penalty_rate"], label="whole GA non penalty rate", color="blue")
axs[1, 1].set_title("None penalty rates")
axs[1, 1].set_xlabel("Generations")
axs[1, 1].set_ylabel("%")
axs[1, 1].legend()

fig.tight_layout()
fig.savefig(dir_anal + "best_fit_and_avg_len.png")
plt.close(fig)


plt.plot(df["generation"]+1, df["mean_fit"], color="blue")
plt.title("Mean of fit of genomes in each generation")
plt.axhline(y=ground_truth, color='r', linestyle='-', label="Ground Truth")
plt.xlabel("Generations")
plt.ylabel("units")
plt.tight_layout()
plt.savefig(dir_anal + "mean_fits.png")
plt.clf()

print(df["best_energy"].min() - ground_truth)
