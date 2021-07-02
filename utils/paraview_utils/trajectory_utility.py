import msgpack


class DesyncError(Exception):
    pass


def load_frame(fhs, fpos, index):
    data = []
    for i in range(len(fhs)):
        fhs[i].seek(fpos[i][index])
        data.append(msgpack.Unpacker(fhs[i], raw=False).unpack())

    time = data[0]["time"]
    dt = data[0]["dt"]
    fibers = []
    bodies = []
    for el in data:
        if el["time"] != time or el["dt"] != dt:
            raise DesyncError
        fibers.extend(el["fibers"][0])
        el.pop("fibers")

    data[0]["fibers"] = fibers
    data[0]["bodies"] = data[0]["bodies"][0]

    return data[0]


def get_frame_sizes(filenames):
    if not filenames:
        return [], []

    unpackers = []
    fhs = []
    for filename in filenames:
        f = open(filename, "rb")
        fhs.append(f)
        unpackers.append(msgpack.Unpacker(f, raw=False))

    fpos = [[] for i in range(len(filenames))]
    while True:
        try:
            for i in range(len(unpackers)):
                fpos[i].append(unpackers[i].tell())
                unpackers[i].skip()

        except msgpack.exceptions.OutOfData:
            fpos[0].pop()
            break

    return fhs, fpos
