import cv2
import mediapipe as mp
import numpy as np
import time, os

# 대화 예시 ########################
# 안녕하세요 /
# 신용 / 대출 / 가능한가요? /
# 네 / 아니요 /           
###################################

def create(word):
    file_path = "classset.txt" # 라벨명

    seq_length = 30 # 윈도우 사이즈 30
    secs_for_action = 30 # 녹화 시간 30초

    # MediaPipe hands model initialize
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1080) # set video widht
    cap.set(4, 720) # set video height

    created_time = int(time.time())
    os.makedirs('dataset', exist_ok=True) # 데이터셋 저장 폴더
    
    action = word
    
    actions = []
    
    data = []
    
    with open(file_path, 'r', encoding="UTF-8") as f:
        actions = f.read().splitlines()
        print(actions)
        
    if action not in actions:
        with open(file_path, 'a', encoding="UTF-8") as f:
                f.write(f"{word}\n")
                id = len(actions)
                                      
        if cap.isOpened():
            ret, img = cap.read()

            img = cv2.flip(img, 1)

            cv2.imshow('img', img)
            cv2.waitKey(3000)

            start_time = time.time()

            while time.time() - start_time < secs_for_action: # 30초 동안 녹화
                ret, img = cap.read()

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img) # 하나씩 프레임을 읽어서 미디어 파이브에 넣어줌
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None: # 결과를 가지고 각도를 뽑아내는 과정
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # 손가락 각도가 이미지 상에서 보이는지 안보이는지를 판단

                        # 손가락 관절사이에 각도를 구하는 코드
                        # Compute angles between joints
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                        v = v2 - v1 # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle) # Convert radian to degree

                        # 라벨(id)을 넣음
                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, id)

                        # 수집한 좌표값들을 펼쳐서 concatenate -> 100개짜리 행렬
                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    break
            
            # numpy array 형태로 변환
            data = np.array(data)
            print(action, data.shape)   
            
            # npy형태로 저장
            np.save(os.path.join('dataset', f'raw_{id}_{action}'), data)

            # Create sequence data
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            print(action, full_seq_data.shape)
            np.save(os.path.join('dataset', f'seq_{id}_{action}'), full_seq_data)
            
            cap.release()
            

    else:
        print("exist class!")

# create("hello")


