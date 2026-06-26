#!/usr/bin/env python3
# encoding: utf-8
# autonomous driving (v2: depth 거리 기반 우선순위 + 통합 상태머신)
#
# 핵심 구조
# - depth 카메라(16UC1, 640x480, RGB와 동일 해상도)로 각 표지까지 실제 거리(m) 측정
# - 매 콜백마다 모든 표지의 거리를 구해 "가장 가까운 표지 하나"를 선택
#   단, 빨간불이 RED_PRIORITY_DISTANCE 이내면 거리 0으로 깔아 최우선
# - main 루프의 단일 상태머신이 그 표지를 소비해 행동 결정
# - 모든 cmd_vel publish는 main 한 곳에서만 발생 (멀티스레드 경합 제거)
#
# ※ "직접 조정" 상수는 실제 로봇에서 시험하며 값을 정하세요.

import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
import sdk.common as common
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup


# ===== 직접 조정하는 상수 =====
# --- depth 토픽 / 거리 측정 ---
DEPTH_TOPIC = '/ascamera/camera_publisher/depth0/image_raw'  # 다르면 이 한 줄만 수정
DEPTH_MM_TO_M = 1000.0          # 16UC1 = 밀리미터 → 미터
DEPTH_PATCH = 2                 # 박스 중심 주변 (2*P+1)x(2*P+1) 영역 중앙값 사용 (노이즈 완화)
DEPTH_MIN_VALID = 0.15          # 이 거리(m) 미만은 측정 실패로 간주 (카메라 최소거리)
DEPTH_MAX_VALID = 4.0           # 이 거리(m) 초과는 무효 (카메라 최대거리)

# --- 빨간불 최우선 ---
RED_PRIORITY_DISTANCE = 0.3     # 빨간불이 이 거리(m) 이내면 거리정렬 1순위로 (직접 조정)

# --- 횡단보도 정지 ---
CROSSWALK_STOP_DISTANCE = 0.4   # 횡단보도가 이 거리(m) 이내면 정지 (직접 조정)
CROSSWALK_DETECT_CONFIRM = 3    # 오검출 방지용 연속 감지 횟수 (직접 조정)
NO_LIGHT_TIMEOUT = 3.0          # 정지 후 이 시간(초) 동안 신호 없으면 통과 (직접 조정)
RED_LOSS_TOLERANCE = 5          # 빨강 깜빡임 허용 카운트 (직접 조정)

# --- 주차 ---
PARK_TRIGGER_DISTANCE = 0.35    # 주차표지가 이 거리(m) 이내면 주차 기동 시작 (직접 조정)
PARK_CONFIRM = 15               # 주차 시작 전 연속 확인 횟수 (직접 조정)

# --- 우회전 (개루프) ---
RIGHT_TURN_CONFIRM = 5          # 'right' 연속 감지 횟수 (직접 조정)
RIGHT_TURN_TRIGGER_DISTANCE = 0.5  # 이 거리(m) 이내로 들어오면 회전 시작 (직접 조정)
RIGHT_TURN_ANGULAR_Z = -0.9     # 회전 각속도 rad/s (음수=우회전) (직접 조정)
RIGHT_TURN_DURATION = 1.5       # 회전 지속 시간(초) (직접 조정)
RIGHT_TURN_LINEAR_X = 0.0       # 회전 중 전진 속도 (0이면 제자리, 직접 조정)

# --- 주행 ---
DRIVE_SPEED = 0.3               # 전진 속도 (직접 조정)
LANE_CENTER_SETPOINT = 100      # 차선 중앙일 때 lane_x 기준값
LANE_TURN_THRESHOLD = 120       # lane_x가 이보다 크면 급커브로 간주

# --- 안전 타임아웃 (상태 stuck 방지) ---
STATE_MAX_DURATION = 8.0        # 어떤 상태에 이 시간(초) 이상 머물면 강제로 주행 복귀

# --- 차량 파라미터 ---
ACKER_WHEELBASE = 0.145
# ==============================


class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True

        self.pid = pid.PID(0.5, 0.01, 0.08)
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()
        self.mecanum_pub.publish(Twist())
        time.sleep(1)

        self.display = True
        self.enter_srv_callback(Trigger.Request(), Trigger.Response())
        request = SetBool.Request()
        request.data = True
        self.set_running_srv_callback(request, SetBool.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False

        # 최신 depth 프레임 (미터 단위 float 배열)
        self.depth_frame = None

        # 거리 우선순위로 뽑힌 "현재 최우선 표지"
        # dict: {'class_name', 'distance', 'box', 'center'} 또는 None
        self.priority_target = None

        # 통합 상태머신
        # 상태: 'LANE_FOLLOW', 'STOP_AT_CROSSWALK', 'TURNING_RIGHT', 'PARKING'
        self.drive_state = 'LANE_FOLLOW'
        self.state_enter_time = time.time()

        # 횡단보도/신호등
        self.count_crosswalk = 0
        self.red_loss_count = 0

        # 우회전
        self.count_right = 0
        self.right_turn_start = 0

        # 주차
        self.count_park = 0
        self.park_done = False

        # 차선 급커브
        self.count_turn = 0
        self.start_turn = False
        self.start_turn_time_stamp = 0

        self.objects_info = []
        self.image_sub = None
        self.object_sub = None
        self.depth_sub = None

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            self.image_sub = self.create_subscription(
                Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
            self.depth_sub = self.create_subscription(
                Image, DEPTH_TOPIC, self.depth_callback, 1)
            self.object_sub = self.create_subscription(
                ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
            try:
                for sub in (self.image_sub, self.depth_sub, self.object_sub):
                    if sub is not None:
                        self.destroy_subscription(sub)
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    def depth_callback(self, ros_image):
        # 16UC1 → 밀리미터. passthrough로 원본 유지 후 미터로 환산은 조회 시점에 수행
        depth_mm = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
        self.depth_frame = depth_mm  # numpy uint16 (H, W)

    # ---- 박스 중심 픽셀의 실제 거리(m) 측정 ----
    def get_distance(self, box):
        if self.depth_frame is None:
            return None
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        h, w = self.depth_frame.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        # 중심 주변 패치의 유효값 중앙값으로 노이즈 완화
        x0, x1 = max(0, cx - DEPTH_PATCH), min(w, cx + DEPTH_PATCH + 1)
        y0, y1 = max(0, cy - DEPTH_PATCH), min(h, cy + DEPTH_PATCH + 1)
        patch = self.depth_frame[y0:y1, x0:x1].astype(np.float32) / DEPTH_MM_TO_M
        valid = patch[(patch >= DEPTH_MIN_VALID) & (patch <= DEPTH_MAX_VALID)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ---- 우회전 / 주차 기동: 단일 step 함수 (상태머신이 매 루프 호출) ----
    def park_action(self):
        # 개루프 주차. 기존 동작 유지. (상태 PARKING 진입 시 1회 실행)
        if self.machine_type == 'MentorPi_Mecanum':
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38 / 0.2)
        elif self.machine_type == 'MentorPi_Acker':
            for lx, az_sign, dt in ((0.15, 1, 3), (0.15, -1, 2), (-0.15, 1, 1.5)):
                twist = Twist()
                twist.linear.x = lx
                twist.angular.z = az_sign * lx * math.tan(-0.5061) / ACKER_WHEELBASE
                self.mecanum_pub.publish(twist)
                time.sleep(dt)
        else:
            twist = Twist(); twist.angular.z = -1.0
            self.mecanum_pub.publish(twist); time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
            twist = Twist(); twist.linear.x = 0.2
            self.mecanum_pub.publish(twist); time.sleep(0.65 / 0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist(); twist.angular.z = 1.0
            self.mecanum_pub.publish(twist); time.sleep(1.5)
        self.mecanum_pub.publish(Twist())

    # ---- 상태 전이 헬퍼 ----
    def set_state(self, new_state):
        if new_state != self.drive_state:
            self.get_logger().info('\033[1;36mSTATE: %s -> %s\033[0m' % (self.drive_state, new_state))
            self.drive_state = new_state
            self.state_enter_time = time.time()

    # ---- 통합 상태머신: 매 루프 1회. twist를 반환 ----
    def update_state_machine(self):
        twist = Twist()
        target = self.priority_target  # 가장 가까운(또는 빨간불 최우선) 표지
        cls = target['class_name'] if target else None
        dist = target['distance'] if target else None

        # 안전: 어떤 상태든 너무 오래 머물면 주행 복귀 (stuck 방지)
        if self.drive_state != 'LANE_FOLLOW' and \
                (time.time() - self.state_enter_time) > STATE_MAX_DURATION:
            self.set_state('LANE_FOLLOW')

        # ---- 상태별 처리 ----
        if self.drive_state == 'PARKING':
            # 주차는 개루프 1회 실행 후 종료. 여기선 정지 유지.
            return Twist()

        if self.drive_state == 'TURNING_RIGHT':
            if (time.time() - self.right_turn_start) < RIGHT_TURN_DURATION:
                if self.machine_type == 'MentorPi_Acker':
                    lx = max(RIGHT_TURN_LINEAR_X, 0.1)
                    twist.linear.x = lx
                    twist.angular.z = lx * math.tan(RIGHT_TURN_ANGULAR_Z) / ACKER_WHEELBASE
                else:
                    twist.linear.x = RIGHT_TURN_LINEAR_X
                    twist.angular.z = RIGHT_TURN_ANGULAR_Z
                return twist
            else:
                self.set_state('LANE_FOLLOW')

        if self.drive_state == 'STOP_AT_CROSSWALK':
            # 정지 유지하며 신호 판정
            if cls == 'green':
                self.set_state('LANE_FOLLOW')
                return self.lane_follow_twist()
            if cls == 'red':
                self.red_loss_count = 0
                return Twist()  # 정지
            # 신호등이 안 보임: 깜빡임 vs 신호없음 구분
            self.red_loss_count += 1
            if (time.time() - self.state_enter_time) > NO_LIGHT_TIMEOUT \
                    and self.red_loss_count > RED_LOSS_TOLERANCE:
                self.set_state('LANE_FOLLOW')   # 신호없는 횡단보도 → 통과
                return self.lane_follow_twist()
            return Twist()  # 판단 보류, 정지 유지

        # ---- LANE_FOLLOW 상태: 최우선 표지에 따라 전이 결정 ----
        if cls == 'red' and dist is not None and dist <= RED_PRIORITY_DISTANCE:
            # 빨간불 최우선 정지 (횡단보도 없이도)
            self.set_state('STOP_AT_CROSSWALK')
            return Twist()

        if cls == 'crosswalk' and dist is not None and dist <= CROSSWALK_STOP_DISTANCE:
            self.count_crosswalk += 1
            if self.count_crosswalk >= CROSSWALK_DETECT_CONFIRM:
                self.count_crosswalk = 0
                self.set_state('STOP_AT_CROSSWALK')
                return Twist()
        else:
            self.count_crosswalk = 0

        if cls == 'park' and dist is not None and dist <= PARK_TRIGGER_DISTANCE:
            self.count_park += 1
            if self.count_park >= PARK_CONFIRM and not self.park_done:
                self.count_park = 0
                self.set_state('PARKING')
                self.mecanum_pub.publish(Twist())
                self.park_done = True
                threading.Thread(target=self.park_action, daemon=True).start()
                return Twist()
        else:
            self.count_park = 0

        if cls == 'right' and dist is not None and dist <= RIGHT_TURN_TRIGGER_DISTANCE:
            self.count_right += 1
            if self.count_right >= RIGHT_TURN_CONFIRM:
                self.count_right = 0
                self.right_turn_start = time.time()
                self.set_state('TURNING_RIGHT')
                return twist  # 다음 루프부터 회전
        else:
            self.count_right = 0

        # 특별한 표지가 없으면 일반 차선 주행
        return self.lane_follow_twist()

    # ---- 차선 추종 twist 계산 ----
    def lane_follow_twist(self):
        twist = Twist()
        twist.linear.x = DRIVE_SPEED
        if self._lane_x is None or self._lane_x < 0:
            self.pid.clear()
            return twist
        lane_x = self._lane_x
        if lane_x > LANE_TURN_THRESHOLD:
            self.count_turn += 1
            if self.count_turn > 5 and not self.start_turn:
                self.start_turn = True
                self.count_turn = 0
                self.start_turn_time_stamp = time.time()
            if self.machine_type != 'MentorPi_Acker':
                twist.angular.z = -0.9
            else:
                twist.angular.z = twist.linear.x * math.tan(-0.9) / ACKER_WHEELBASE
        else:
            self.count_turn = 0
            if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                self.start_turn = False
            if not self.start_turn:
                self.pid.SetPoint = LANE_CENTER_SETPOINT
                self.pid.update(lane_x)
                if self.machine_type != 'MentorPi_Acker':
                    twist.angular.z = common.set_range(self.pid.output, -0.1, 0.1)
                else:
                    twist.angular.z = twist.linear.x * math.tan(
                        common.set_range(self.pid.output, -0.1, 0.1)) / ACKER_WHEELBASE
            elif self.machine_type == 'MentorPi_Acker':
                twist.angular.z = 0.15 * math.tan(-0.5061) / ACKER_WHEELBASE
        return twist

    def main(self):
        self.get_logger().info('\033[1;33m%s\033[0m' % "self_driving main start")
        self._lane_x = -1

        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                continue

            result_image = image.copy()
            if self.start:
                binary_image = self.lane_detect.get_binary(image)
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())
                self._lane_x = lane_x

                # 통합 상태머신이 모든 행동을 결정하고 단일 publish
                twist = self.update_state_machine()
                self.mecanum_pub.publish(twist)

                # 디버그 박스
                if self.objects_info:
                    for i in self.objects_info:
                        cls_id = self.classes.index(i.class_name)
                        plot_one_box(
                            i.box, result_image,
                            color=colors(cls_id, True),
                            label="{}:{:.2f}".format(i.class_name, i.score))
            else:
                time.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    # ---- 객체 감지 콜백: 거리 측정 + 최우선 표지 선택 ----
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if not self.objects_info:
            self.priority_target = None
            return

        candidates = []
        for i in self.objects_info:
            dist = self.get_distance(i.box)
            if dist is None:
                continue
            center = (int((i.box[0] + i.box[2]) / 2), int((i.box[1] + i.box[3]) / 2))
            # 정렬 키: 기본은 실제 거리. 단 빨간불이 임계 이내면 0으로 최우선.
            sort_key = dist
            if i.class_name == 'red' and dist <= RED_PRIORITY_DISTANCE:
                sort_key = 0.0
            candidates.append({
                'class_name': i.class_name,
                'distance': dist,
                'box': i.box,
                'center': center,
                'sort_key': sort_key,
            })

        if not candidates:
            self.priority_target = None
            return

        candidates.sort(key=lambda c: c['sort_key'])
        self.priority_target = candidates[0]


def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()