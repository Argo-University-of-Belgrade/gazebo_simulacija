#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MoveController(Node):
    def __init__(self):
        super().__init__('move_controller')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move_sequence)  # 10Hz timer
        self.start_time = time.time()
        self.phase = 0  # 0=forward, 1=turn, 2=stop

    def move_sequence(self):
        twist = Twist()
        current_time = time.time()
        elapsed = current_time - self.start_time

        if self.phase == 0:  # Move forward
            twist.linear.x = 0.5  # Forward speed (m/s)
            if elapsed > 5.0:  # After 5 seconds
                self.phase = 1
                self.start_time = current_time  # Reset timer for next phase
        elif self.phase == 1:  # Turn right
            twist.angular.z = -0.5  # Right turn (rad/s)
            if elapsed > 2.0:  # Turn for 2 seconds
                self.phase = 2
        else:  # Stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.timer.cancel()  # Stop the timer

        self.publisher.publish(twist)
        self.get_logger().info(f'Phase: {self.phase}, Linear: {twist.linear.x}, Angular: {twist.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = MoveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send stop command before exiting
        twist = Twist()
        node.publisher.publish(twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
