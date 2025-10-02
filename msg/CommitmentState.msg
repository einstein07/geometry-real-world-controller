# CommitmentState.msg
string robot_id                     # unique robot name/id
builtin_interfaces/Time stamp       # timestamp of this message
uint64 seq                          # sequence counter (increments each publish)
int32 commitment                    # 0 = uncommitted, 1 = Target1, 2 = Target2
float32 quality                     # optional confidence value [0.0, 1.0] (optional)
