# 三旋翼 sim2real 域随机化与噪声策略

本说明文档面向科研与工程部署场景，系统化描述三旋翼平台在仿真—实装迁移（sim2real）过程中所使用的域随机化与噪声建模。所有设定均围绕 `TrirotorCfg` 与任务 `PositionSetpointTaskSim2RealEndToEnd`，致力于在训练阶段覆盖真实世界的姿态、动力学与感知不确定性。

## 1. 初始状态域随机化

设机器人在 episode 开始时的状态向量为
\[
\mathbf{s}_0 = [x,\; y,\; z,\; \phi,\; \theta,\; \psi,\; 1,\; v_x,\; v_y,\; v_z,\; \omega_x,\; \omega_y,\; \omega_z]^\top,
\]
其中 $\phi,\theta,\psi$ 分别表示机体系的滚转、俯仰与偏航角。该向量按照分量独立的均匀分布采样：
\[
\mathbf{s}_0 \sim \mathcal{U}(\mathbf{s}_{\min},\; \mathbf{s}_{\max}),
\]
并取
\[
\mathbf{s}_{\min} = [-0.7,-0.7,-0.7,-\tfrac{\pi}{6},-\tfrac{\pi}{6},-\pi,1,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5]^\top,
\]
\[
\mathbf{s}_{\max} = [0.7,0.7,0.7,\tfrac{\pi}{6},\tfrac{\pi}{6},\pi,1,0.5,0.5,0.5,0.5,0.5,0.5]^\top.
\]
如此可以同时拓展初始位置、姿态以及线/角速度的覆盖面，确保策略在大偏差起飞、扰动启动等场景中仍能稳定收敛。

此外，机器人资产层面还定义了比例化的状态窗口 $\mathbf{r}_{\min}$、$\mathbf{r}_{\max}$，用于约束随机生成位置与姿态的工作空间范围，避免极端碰撞或不可恢复的初值。

## 2. 执行器与机构初始化

三对倾转伺服的初始状态记为
\[
\boldsymbol{\delta}_0 = [\delta_1,\delta_2,\delta_3,\dot{\delta}_1,\dot{\delta}_2,\dot{\delta}_3]^\top,
\]
其采样分布为
\[
\delta_i \sim \mathcal{U}(-0.05,\; 0.05),\quad \dot{\delta}_i \sim \mathcal{U}(-0.05,\;0.05).
\]
该设定促使控制器面对运动学不对称或关节偏置时仍具备快速调姿能力。伺服在仿真中使用位置模式，额外配置了等效刚度 $k_s = 4.0$ 与阻尼 $c_s = 0.2$，对应真实伺服的闭环行为。

在推进器层面，控制分配模块采用连续时间的推力—力矩模型。若 $\omega$ 为电机角速度，推力系数 $C_T$ 固定为 $2\times10^{-5}$，则单电机最大静态推力满足
\[
F_{\max} = C_T \, \omega_{\max}^2 \approx 45\;\text{N},
\]
并配合上升/下降时间常数 $\tau_{\uparrow}=0.0125\,\text{s}$、$\tau_{\downarrow}=0.025\,\text{s}$ 模拟真实转速滞后。

## 3. 动力学扰动与环境随机化

### 3.1 瞬时力/力矩扰动

在每个时间步 $t$，采样伯努利变量 $b_t \sim \mathrm{Bernoulli}(p_d)$，其中 $p_d = 0.02$。若 $b_t = 1$，则对机体施加六维扰动矢量
\[
\mathbf{d}_t = [f_x,f_y,f_z,\tau_x,\tau_y,\tau_z]^\top,
\]
其中
\[
f_i \sim \mathcal{U}(-10^{-3},10^{-3})\;\text{N},\qquad \tau_i \sim \mathcal{U}(-4\times10^{-5},4\times10^{-5})\;\text{N·m}.
\]
该机制覆盖了微风脉冲、螺旋桨尾流、轻量碰撞等突发情况。

### 3.2 稳态风场

空气动力配置引入定向风场：
\[
\mathbf{v}_{\text{wind}} = w\,\hat{\mathbf{e}}_x, \qquad w \sim \mathcal{U}(0,20)\;\text{m/s}.
\]
给定参考面积 $A = 0.005\,\text{m}^2$、空气密度 $\rho = 1.225\,\text{kg/m}^3$，则对机体施加的平均阻力幅值约为
\[
\|\mathbf{F}_{\text{drag}}\| \approx \tfrac{1}{2}\,\rho\,A\,C_D\,w^2,
\]
其中阻力系数 $C_D = C_{d0} + C_{d\alpha^2} \, \alpha^2$，升力系数以线性模型 $C_L = C_{L\alpha}\,\alpha$ 估计，$\alpha$ 为攻角。通过随机化 $w$，策略能够学习在不同风速下保持航迹与姿态。

## 4. 观测噪声模型

仿真感知管线在输出到策略前引入零均值高斯噪声，用以逼近里程计、IMU 与状态估计器的误差。对于机器人位置 $\mathbf{p}$、欧拉角 $\boldsymbol{\eta}$、线速度 $\mathbf{v}$、角速度 $\boldsymbol{\omega}$，分别定义
\[
\tilde{\mathbf{p}} = \mathbf{p} + \boldsymbol{\epsilon}_p,\quad \boldsymbol{\epsilon}_p \sim \mathcal{N}(\mathbf{0}, 10^{-6}\,\mathbf{I}),
\]
\[
\tilde{\boldsymbol{\eta}} = \boldsymbol{\eta} + \boldsymbol{\epsilon}_{\eta},\quad \boldsymbol{\epsilon}_{\eta} \sim \mathcal{N}(\mathbf{0}, (3.05\times10^{-3})^2\,\mathbf{I}),
\]
\[
\tilde{\mathbf{v}} = \mathbf{v} + \boldsymbol{\epsilon}_v,\quad \boldsymbol{\epsilon}_v \sim \mathcal{N}(\mathbf{0}, (2\times10^{-3})^2\,\mathbf{I}),
\]
\[
\tilde{\boldsymbol{\omega}} = \boldsymbol{\omega} + \boldsymbol{\epsilon}_{\omega},\quad \boldsymbol{\epsilon}_{\omega} \sim \mathcal{N}(\mathbf{0}, (10^{-3})^2\,\mathbf{I}).
\]
姿态在下游通过欧拉角—四元数转换后映射为六维旋转表示 $\mathbf{r}_6 \in \mathbb{R}^6$，从而保持网络输入的连续性。若仿真环境提供风速标量 $w$，则观测拓展一维以形成
\[
\mathbf{o} = [\tilde{\mathbf{p}} - \mathbf{p}_{\text{target}},\; \mathbf{r}_6,\; \tilde{\mathbf{v}},\; \tilde{\boldsymbol{\omega}},\; w]^\top.
\]

## 5. 动作约束与正则化

### 5.1 动作缩放

策略输出 $\mathbf{a} \in [-1,1]^6$ 经剪裁后按照仿真—硬件一致的仿射缩放：
\[
\mathbf{u} = \frac{\mathbf{a}_c}{2} \odot (\mathbf{u}_{\max} - \mathbf{u}_{\min}) + \frac{\mathbf{u}_{\max} + \mathbf{u}_{\min}}{2},
\]
其中 $\mathbf{a}_c = \mathrm{clip}(\mathbf{a}, -1, 1)$，前三维对应伺服角限制 $\mathbf{u}_{\min}^{\text{servo}} = [-0.35,-0.35,-0.35]$ rad、$\mathbf{u}_{\max}^{\text{servo}} = [0.35,0.35,0.35]$ rad，后三维对应推力范围 $[0,45]$ N。该映射直接约束策略生成的指令满足硬件可行性。

### 5.2 悬停偏差与能耗惩罚

根据观测得到的机器人质量 $m$ 与电机数量 $n_m=3$，估算每个电机的静态悬停推力
\[
F_{\text{hover}} = \frac{m g}{n_m}.
\]
将动作中的推力分量记为 $\mathbf{f}$，其偏差度量为
\[
\boldsymbol{\delta}_f = \frac{\max(\mathbf{f}, \mathbf{0}) - F_{\text{hover}}}{F_{\text{hover}}}.
\]
奖励函数中加入平滑惩罚项
\[
R_{\text{hover}} = -\lambda_h \sum_i \left( e^{-4 \delta_{f,i}^2} - 1 \right), \quad \lambda_h = 0.05,
\]
以及能耗正则
\[
R_{\text{energy}} = -\lambda_e \sum_i f_i^{3/2}, \quad \lambda_e = 10^{-3},
\]
以鼓励策略在外界扰动下仍保持能效和推力稳定。

### 5.3 动作平滑

对连续时间步的指令差分 $\Delta \mathbf{u}_t = \mathbf{u}_t - \mathbf{u}_{t-1}$ 施加指数惩罚：
\[
R_{\text{smooth}} = -\lambda_s \sum_i \left( e^{-6 (\Delta u_{t,i})^2} - 1 \right), \quad \lambda_s = 0.1.
\]
该项抑制由随机化引起的高频震荡，有助于保护伺服与推进系统。

## 6. 诊断与监控

为了分析随机化后的控制行为，系统提供周期性日志：当仿真步数满足 $t \equiv 0 \pmod{100}$ 时，记录第一个环境中电机的实测推力、指令推力以及目标动作 $\mathbf{u}_t$ 与 $F_{\text{hover}}$。该监控机制支持对能耗、推力分配与扰动恢复效果进行事后量化。

---

通过以上多层级的域随机化——涵盖初始条件、执行机构、外界扰动、观测噪声与动作正则——训练到的策略能够在更广泛的非理想条件下保持姿态与位置控制性能，从而显著提升三旋翼平台的 sim2real 迁移鲁棒性。
