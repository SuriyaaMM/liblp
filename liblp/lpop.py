import enum

class lpop(enum.Enum):
    encode = "encode"
    decode = "decode"
    add = "addition"
    sub = "subtract"
    mul = "multiplication"
    div = "division"
    matmul = "matmul"
    mv_w4a16 = "mv_w4a16"
    mv_w8a16 = "mv_w4a16"
    mm_a32 = "mm_a32"
    mm_a16 = "mm_a16"