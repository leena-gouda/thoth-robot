# Published by thoth_vision on /visitor/attributes
std_msgs/Header header
bool face_detected
string age_group            # child | teen | adult | senior | unknown
float32 age_confidence      # 0.0 - 1.0
string emotion              # happy | neutral | sad | angry | surprised | unknown
float32 emotion_confidence  # 0.0 - 1.0
