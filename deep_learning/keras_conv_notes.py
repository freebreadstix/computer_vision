
# filters: number of filters increase as layers go

# kernel size: size of kernel, dims must be odd, normally square
# Typical vals: (7, 7), (5, 5), (3, 3), (1, 1)
# If image larger 128 * 128 start w/ filters bigger than (3, 3), else/when reduced go (3, 3) or smaller

# stride: distance kernel moves each time convolution applied, 2d tuple representing dimensions
# bigger strides equal dimension reduction

# padding: whether image padded with zeros on border so dimensions maintained during convolution or not