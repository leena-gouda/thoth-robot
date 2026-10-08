# Published by thoth_vision/visitor_perception_node on /visitor/attributes
# One message per analysed frame, describing the PRIMARY visitor (the closest face, kept "sticky").
# Published even when nobody is in view (face_detected = false), so subscribers can detect a dead node.

std_msgs/Header header

bool face_detected
uint32 visitor_id           # tracker ID of the primary visitor; changes when a new person becomes primary (0 = none)
uint8 num_faces             # faces currently in view (incl. bystanders)

string age_group            # child | teen | adult | senior | unknown
float32 age_confidence      # 0.0 - 1.0

string emotion              # happy | sad | neutral | excited | unknown   (bored: planned, needs arousal model)
float32 emotion_confidence  # 0.0 - 1.0
