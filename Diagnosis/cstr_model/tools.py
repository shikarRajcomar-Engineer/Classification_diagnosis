
import matplotlib.pyplot as plt


class PID:
    """
    Simple implementation of a PID controller. 
    """
    def __init__(self, setpoint, K_p,tau_i=None, u_bias=0, sample_dt=0.1, limits=(0,1)):
        """
        Initialise controller

        Args:
            setpoint (float): PV variable setpoint to control
            K_p (float): Proportional gain 
            tau_i (float): Controller lag for integral
            u_bias (float): Bias for controller output (typically last value before it is set to auto)
            sample_dt (float): dt intervals when controller output should be updated
            limits (tuple): Saturation limits for controller output as (min, max)
        """
        
        self.u_bias = u_bias
        self.K_p = K_p
        self.tau_i = tau_i
        self.setpoint = setpoint
        self.I=0
        self.u=u_bias
        self.prev_t = 0
        self.sample_dt=sample_dt
        self.limits = limits
        self.store=[]

    def __call__(self,PV,t):
        """
        Compute controller output

        Args:
            PV (float): Process variable value
            t (float): current time

        Returns:
            float: controller output
        """
        
        # Calculate dt and check if controller output should be updated
        dt = t-self.prev_t
        if dt <self.sample_dt:
            return self.u
        
        # Calculate error and controller components (P and I)
        error = (self.setpoint-PV)
        P = error*self.K_p
        if self.tau_i:
            self.I += dt*error/self.tau_i

        # Calculate controller output and bound in by the limits
        u = self.u_bias + P + self.I


        lim_high = self.limits[1]
        lim_low = self.limits[0]

        # implement anti-reset windup
        if u < lim_low or u > lim_high:
            if self.tau_i:
                self.I -=  dt*error/self.tau_i
            # clip output
            u = max(lim_low,min(u,lim_high))

    
        # Update variables and return result
        self.u = u
        self.prev_t = t
        self.store.append([t,u])
        return u

    def plot_results(self):
        x = [item[0] for item in self.store]
        y = [item[1] for item in self.store]
        plt.plot(x,y)


if __name__ == "__main__":
    

    t= 0
    dt = 1
    u = 100
    PV=150

    pid = PID(setpoint=150, K_p=0.1, tau_i=None,limits=(0,100),u_bias=u, sample_dt=dt)

    store = []

    while t<100:

        if t ==10:
            pid.setpoint = 100

        u = pid(PV, t)
        PV = PV*u/100

        print(u, PV)
        store.append([t,PV])

        t+=dt

    x = [item[0] for item in store]
    y = [item[1] for item in store]
    plt.plot(x,y)

    pid.plot_results()


    plt.legend(['PV','Controller output'])
    plt.show()
    print('Fin!')

    