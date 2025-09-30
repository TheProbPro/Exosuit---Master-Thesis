from dynamixel_sdk import *
import time
import numpy as np
import sys

#from DynamixelSDK.python.tests.protocol_combined import TORQUE_ENABLE

""" This class should interface the control of the motors using the Dynamixel SDK."""
class Motors:
    POS_CONTROL = 3
    VEL_CONTROL = 1
    CUR_CONTROL = 0

    CONTROL_DICT = {"pos": POS_CONTROL,
                    "vel": VEL_CONTROL,
                    "cur": CUR_CONTROL}
    """ Check which COM-port the motor is connected to by using the command python "-m serial.tools.list_ports" in the terminal."""
    def __init__(self, port = "COM3", baudrate = 3000000):
        # Set memeber variables
        self.port = port
        self.baudrate = baudrate
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(2.0)
        self.ping_num = 50
        
        # Open port and set baudrate
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            raise Exception("Failed to open the port")
        if self.portHandler.setBaudRate(self.baudrate):
            print("Succeeded to change the baudrate")
        else:
            raise Exception("Failed to set the baudrate")
        
        # Get motor IDs
        self.motor_ids = self.get_motor_ids()
        self.num_motors = len(self.motor_ids)
        if self.num_motors == 0:
            raise Exception("No motors found")
        print(f"Found {self.num_motors} motors with IDs: {self.motor_ids}")

        # TODO: Set up groupsyncreads if there is more than one motor.

        # Set initial motor parameters
        self.torque_enabled = np.zeros(self.num_motors, dtype = bool)
        self.control_mode = np.zeros(self.num_motors, dtype = int) # 0: current, 1: velocity, 3: position

        # initialize all motors to position control mode
        self.set_cont_mode("pos")

    """ Enable torque for all motors """
    def enable_torque(self):
        for i in range(self.num_motors):
            if not self.torque_enabled[i]:
                self.write(self.motor_ids[i], 64, 1, 1)
                self.torque_enabled[i] = True
            else:
                print(f"Motor {self.motor_ids[i]} torque is already enabled.")
        print("All motors torque enabled.")

    """ Disable torque for all motors """
    def disable_torque(self):
        for i in range(self.num_motors):
            if self.torque_enabled[i]:
                self.write(self.motor_ids[i], 64, 1, 0)
                self.torque_enabled[i] = False
            else:
                print(f"Motor {self.motor_ids[i]} torque is already disabled.")
        print("All motors torque disabled.")

    """ Ping all possible motor IDs and return a list of the IDs that respond. """
    def get_motor_ids(self):
        motor_ids = []
        for i in range(self.ping_num):
            dxl_model_number, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, i)
            if dxl_comm_result != COMM_SUCCESS:
                n=0
            elif dxl_error != 0:
                n=0
            else:
                motor_ids.append(i)
        return motor_ids

    """ Write commands to the motor. motor_id: ID of the motor, add_write: address to write to, byte_num: number of bytes to write (1, 2 or 4), comm: command to write"""
    def write(self, motor_id, add_write, byte_num, comm):
        assert (byte_num in [1,2,4]), "the writting byte should be one of [1, 2, 4]"
        comm = int(comm)
        if (byte_num == 1):
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, add_write, comm)
        elif (byte_num == 2):
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, add_write, comm)
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, add_write, comm)
        
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    """ Read data from the motor. motor_id: ID of the motor, add_read: address to read from, byte_num: number of bytes to read (1, 2 or 4) """
    def read(self, motor_id, add_read, byte_num):
        assert (byte_num in [1,2,4]), "the reading byte should be one of [1, 2, 4]"
        if (byte_num == 1):
            cl_dxl, cl_dxl_comm_result, cl_dxl_error = self.packetHandler.read1ByteTxRx(self.packetHandler, motor_id, add_read)
        elif (byte_num == 2):
            cl_dxl, cl_dxl_comm_result, cl_dxl_error = self.packetHandler.read2ByteTxRx(self.packetHandler, motor_id, add_read)
        else:
            cl_dxl, cl_dxl_comm_result, cl_dxl_error = self.packetHandler.read4ByteTxRx(self.packetHandler, motor_id, add_read)

        if cl_dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(cl_dxl_comm_result))
        elif cl_dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(cl_dxl_error))
        else:
            return cl_dxl

    """ Set control mode for all motors. mode: "pos", "vel" or "cur" """
    def set_cont_mode(self, mode = "pos"):
        for i in range(self.num_motors):
            if self.control_mode[i] != self.CONTROL_DICT[mode]:
                self.write(self.motor_ids[i], 11, 1, self.CONTROL_DICT[mode])
                self.control_mode[i] = self.CONTROL_DICT[mode]
        print(f"All motors set to {mode} control mode.")

    def get_position(self):
        for i in range(self.num_motors):
            self.now_pos[i] = self.read(self.motor_ids[i], 132, 4)
        return self.now_pos
    
    #TODO: Maybe implement get_velocity and get_current methods as well.
    
    """ Control the motor position. goal_position: desired position in ticks (0-4095) """
    def control_position(self, goal_position):
        for i in range(self.num_motors):
            assert (self.control_mode[i] == self.POS_CONTROL), f"Motor {self.motor_ids[i]} is not in position control mode."
            self.write(self.motor_ids[i], 116, 4, goal_position)
        

    """ Close the port """
    def close(self):
        self.portHandler.closePort()
        print("Port closed.")


if __name__ == "__main__":
    motors = Motors()
    # Test torque enable
    motors.enable_torque()

    #print(motors.get_position())
    # wait for 2 seconds
    #time.sleep(2)
    # Test position control
    motors.control_position(2048)
    time.sleep(2)

    # disable torque
    motors.disable_torque()

    
    print("Motors initialized successfully")

    motors.close()
    print("Program ended")
