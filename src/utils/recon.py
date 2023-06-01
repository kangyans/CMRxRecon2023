from sigpy.mri.app import EspiritCalib, SenseRecon


def espirit_sense(k, calib_size=24, espirit_iter=12, sense_iter=12):
    sensitivity = EspiritCalib(k, calib_width=calib_size, crop=0.0,
                               max_iter=espirit_iter, show_pbar=False).run()
    img = SenseRecon(k, sensitivity, max_iter=sense_iter,
                     show_pbar=False).run()
    return img
