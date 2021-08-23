import torchio as tio


class HomogeniseLaterality(tio.SpatialTransform):
    def __init__(self, include, from_laterality, axes):
        super().__init__(include=include)
        self.from_laterality = from_laterality
        self.axes = axes

    def apply_transform(self, subject):
        transformed = subject
        if subject["laterality"] == self.from_laterality:
            transform = tio.Flip(**self.add_include_exclude(
                {"axes": _ensure_axes_indices(subject, self.axes)}))
            transformed = transform(subject)
        return transformed

    def add_include_exclude(self, kwargs):
        kwargs['include'] = self.include
        kwargs['exclude'] = self.exclude
        return kwargs

    @staticmethod
    def is_invertible():
        return True

    def inverse(self):
        return self


def _ensure_axes_indices(subject, axes):
    if any(isinstance(n, str) for n in axes):
        subject.check_consistent_orientation()
        image = subject.get_first_image()
        axes = sorted(3 + image.axis_name_to_index(n) for n in axes)
    return axes