from crazyflieServer import TrackingObject
from crazyflieServer import CrazyflieServer
from demoRun import run_demo

# Main function 
def main():
    pass
    crazyflies = [
        #TrackingObject( "radio://0/80/2M/E7E7E7E7E7", "crazyflie7" ),
        TrackingObject( "radio://0/80/2M/E7E7E7E7E8", "crazyflie8" ),
        #TrackingObject( "radio://0/80/2M/E7E7E7E7E9", "crazyflie9" ),
    ]
    vicon_ip = "192.168.0.39"
    crazyflieServer = CrazyflieServer( crazyflies, vicon_ip )
    run_demo( crazyflieServer )
    crazyflieServer.close()

if __name__ == "__main__":
    main()