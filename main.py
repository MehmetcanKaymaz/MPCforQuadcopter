import numpy as np
from quad_model import QuadModel
from mpc_controller import MPCController
from linear_mpc_controller import LMPCController
from cascade_MPC import CMPCController
from cascade_NMPC import CNMPCController
from nonlinear_mpc import NMPC
from simple_traj import simple_traj
import matplotlib.pyplot as plt
import time

model_l = QuadModel()
model_hnl = QuadModel()
model_c = QuadModel()
model_n = QuadModel()
model_nsq = QuadModel()
model_cn = QuadModel()

A_init, B_init = model_l.get_A_B()
Ac = np.zeros((3, 3))
B_rate = model_c.get_B_rate()
B_att = model_c.get_B_att()

linear_controller = LMPCController(A=A_init, B=B_init)
half_nonlinear_controller = MPCController()
cascate_mpc_controller = CMPCController(A=Ac, B_rate=B_rate, B_att=B_att)
nonlinear_controller = NMPC()
nonlinear_sq_controller = NMPC(optimizer='sqpmethod')
cascate_nonlinear_controller = CNMPCController(
    optimizer='ipopt', A=Ac, B_rate=B_rate)

planner = simple_traj()
planner2 = simple_traj()
planner3 = simple_traj()
planner4 = simple_traj()
planner5 = simple_traj()
planner6 = simple_traj()

sim_dt = .01
sim_T = 4.5
sim_N = int(sim_T/sim_dt)

t = np.linspace(0, sim_T, sim_N)

linear_freq_arr = []
linear_min_t = 100
linear_max_t = 0
half_nonlinear_freq_arr = []
half_nonlinear_min_t = 100
half_nonlinear_max_t = 0

cascate_freq_arr = []
cascate_max_t = 0
cascate_min_t = 100

nonlinear_freq_arr = []
nonlinear_max_t = 0
nonlinear_min_t = 100

nonlinear_sq_freq_arr = []
nonlinear_sq_max_t = 0
nonlinear_sq_min_t = 100

cascate_nonlinear_sq_freq_arr = []
cascate_nonlinear_sq_max_t = 0
cascate_nonlinear_sq_min_t = 100

for t_curr in t:

    # Cascate NonLinear sq
    x = model_cn.x
    traj = planner6.sample_cascate_nmpc()
    t1 = time.time()
    u = cascate_nonlinear_controller.run_controller(x=x, xr=traj)
    t2 = time.time()
    delta_t = t2-t1
    cascate_nonlinear_sq_freq_arr.append(delta_t)
    if delta_t > cascate_nonlinear_sq_max_t:
        cascate_nonlinear_sq_max_t = delta_t
    if delta_t < cascate_nonlinear_sq_min_t:
        cascate_nonlinear_sq_min_t = delta_t
    x = model_cn.step(u=u, traj=traj)

    # NonLinear sq
    x = model_nsq.x
    traj = planner5.sample_nmpc()
    t1 = time.time()
    u = nonlinear_sq_controller.run_controller(x0=x, ref_states=traj)
    t2 = time.time()
    delta_t = t2-t1
    nonlinear_sq_freq_arr.append(delta_t)
    if delta_t > nonlinear_sq_max_t:
        nonlinear_sq_max_t = delta_t
    if delta_t < nonlinear_sq_min_t:
        nonlinear_sq_min_t = delta_t
    x = model_nsq.step(u=u, traj=traj)

    # NonLinear ipopt
    x = model_n.x
    traj = planner4.sample_nmpc()
    t1 = time.time()
    u = nonlinear_controller.run_controller(x0=x, ref_states=traj)
    t2 = time.time()
    delta_t = t2-t1
    nonlinear_freq_arr.append(delta_t)
    if delta_t > nonlinear_max_t:
        nonlinear_max_t = delta_t
    if delta_t < nonlinear_min_t:
        nonlinear_min_t = delta_t
    x = model_n.step(u=u, traj=traj)

    # Linear
    x = model_l.x
    traj = planner.sample()
    t1 = time.time()
    u = linear_controller.run_controller(x=x, xr=traj)
    t2 = time.time()
    delta_t = t2-t1
    linear_freq_arr.append(delta_t)
    if delta_t > linear_max_t:
        linear_max_t = delta_t
    if delta_t < linear_min_t:
        linear_min_t = delta_t
    x = model_l.step(u=u, traj=traj)

    # Cascate
    x = model_c.x
    traj = planner3.sample_cascate()
    t1 = time.time()
    B_att = model_c.get_B_att()
    u = cascate_mpc_controller.run_controller(B_att=B_att, x=x, xr=traj)
    t2 = time.time()
    delta_t = t2-t1
    cascate_freq_arr.append(delta_t)
    if delta_t > cascate_max_t:
        cascate_max_t = delta_t
    if delta_t < cascate_min_t:
        cascate_min_t = delta_t
    x = model_c.step(u=u, traj=traj)

    # Half Nonlinear
    x = model_hnl.x
    traj = planner2.sample()
    t1 = time.time()
    A, B = model_hnl.get_A_B()
    u = half_nonlinear_controller.run_controller(A=A, B=B, x=x, xr=traj)
    t2 = time.time()
    delta_t = t2-t1
    half_nonlinear_freq_arr.append(delta_t)
    if delta_t > half_nonlinear_max_t:
        half_nonlinear_max_t = delta_t
    if delta_t < half_nonlinear_min_t:
        half_nonlinear_min_t = delta_t
    x = model_hnl.step(u=u, traj=traj)

# model.vis_result()
linear_states, linear_u, traj = model_l.get_all_data()
half_nonlinear_state, half_nonlinear_u, traj = model_hnl.get_all_data()
cascate_state, cascate_u, c_traj = model_c.get_all_data()
nonlinear_states, nonlinear_u, n_traj = model_n.get_all_data()
nonlinear_sq_states, nonlinear_sq_u, nsq_traj = model_nsq.get_all_data()
cascate_nonlinear_states, cascate_nonlinear_u, cascate_n_traj = model_cn.get_all_data()

linear_err = model_l.traj_err
half_nonlinear_err = model_hnl.traj_err
cascate_err = model_c.traj_err
nonlinear_err = model_n.traj_err
nonlinear_sq_err = model_nsq.traj_err
cascate_nonlinear_err = model_cn.traj_err

print("Linear MPC tracking error:", linear_err)
print("Half nonlinear MPC tracking error:", half_nonlinear_err)
print("Cascate MPC tracking error:", cascate_err)
print("Nonlinear MPC tracking error:", nonlinear_err)
print("Nonlinear SQ MPC tracking error:", nonlinear_sq_err)
print("Cascate Nonlinear MPC tracking error:", cascate_nonlinear_err)

print("Linear MPC average_t:{} max_t:{} min_t{}".format(
    np.mean(np.array(linear_freq_arr)), linear_max_t, linear_min_t))
print("Half nonlinear average_t:{} max_t:{} min_t:{}".format(np.mean(
    np.array(half_nonlinear_freq_arr)), half_nonlinear_max_t, half_nonlinear_min_t))
print("Cascate MPC average_t:{} max_t:{} min_t:{}".format(np.mean(
    np.array(cascate_freq_arr)), cascate_max_t, cascate_min_t))
print("Nonlinear MPC average_t:{} max_t:{} min_t{}".format(
    np.mean(np.array(nonlinear_freq_arr)), nonlinear_max_t, nonlinear_min_t))
print("Nonlinear SQ MPC average_t:{} max_t:{} min_t{}".format(
    np.mean(np.array(nonlinear_sq_freq_arr)), nonlinear_sq_max_t, nonlinear_sq_min_t))
print("Cascade Nonlinear SQ MPC average_t:{} max_t:{} min_t{}".format(
    np.mean(np.array(cascate_nonlinear_sq_freq_arr)), cascate_nonlinear_sq_max_t, cascate_nonlinear_sq_min_t))

state_names = ["phi", "theta", "psi", "p", "q", "r"]

t1 = np.linspace(
    0, len(linear_states[:, 0].tolist())*0.001, len(linear_states[:, 0].tolist()))
t2 = np.linspace(
    0, len(traj[:, 0].tolist())*0.01, len(traj[:, 0].tolist()))

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot(t1, linear_states[:, i], label=state_names[i]+"_linear")
    plt.plot(t1, nonlinear_states[:, i], label=state_names[i]+"_nonlinear")
    plt.plot(t1, cascate_nonlinear_states[:, i],
             label=state_names[i]+"_cascade_nonlinear")
    plt.plot(t1, nonlinear_sq_states[:, i],
             label=state_names[i]+"_nonlinear_sq")
    plt.plot(t1, half_nonlinear_state[:, i],
             label=state_names[i]+"_half_nonlinear")
    plt.plot(t1, cascate_state[:, i], label=state_names[i]+"_cascate")
    plt.plot(t2, traj[:, i], label=state_names[i]+"_ref")
    plt.legend()
    if i < 3:
        plt.xlabel("time(s)")
        plt.ylabel("angle(rad)")
    if i > 3:
        plt.xlabel("time(s)")
        plt.ylabel("angular rate(rad/s)")

plt.show()
