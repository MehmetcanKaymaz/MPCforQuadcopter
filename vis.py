import numpy as np
import matplotlib.pyplot as plt


x = ["Linear MPC", "Approximate Nonlinear MPC", "Approximate Cascade NonLinear MPC",
     "Nonlinear MPC(ipopt)", "Nonlinear MPC(SQ)", "Cascade Nonlinear MPC(ipopt)", "Cascade Nonlinear MPC(SQ)"]

y1 = [100.96, 49.46, 51.86, 8.67, 8.67, 10.37, 10.37]
y2 = [0.008, 0.008, 0.015, 0.19, 0.15, 0.14, 0.1]


fig, ax = plt.subplots()
ax2 = ax.twinx()

X_axis = np.arange(len(x))

ax.bar(X_axis-.2, y1, .4,  color='blue')
ax.set_ylabel('Tracking Error(rad)')
ax2.bar(X_axis+.2, y2, .4, color='darkorange')
ax2.set_ylabel('Mean Computation Time(s)')
plt.xticks(X_axis, x)
fig.autofmt_xdate()
ax.legend(['Tracking Error(rad)'], loc=(.85, .9))
ax2.legend(['Mean Computation Time(s)'], loc=1)

for i in X_axis:
    ax.text(i-.2, y1[i], y1[i], ha='center')
    plt.text(i+.2, y2[i], y2[i], ha='center')
ax.set_xlabel('Model Predictive Controllers', fontsize=20)
ax.set_title('Quadcopter Attitude Control', fontsize=25)
plt.show()
