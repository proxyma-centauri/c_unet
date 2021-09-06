import torchio as tio


class HomogeniseLaterality(tio.Flip):
    """
    Transform inheriting from Flip (SpatialTransform).
    Flips the subject along some axis when the 'laterality'
    property of the subject meets a certain pattern

    Args:
        - from_laterality (str): Pattern to match. Defaults to "left"
        - axes (str): Axes to flip on. Defaults to L, meaning the left-right axis.
        - **kwargs: other attributes, like include or exclude.
    """
    def __init__(self, from_laterality="left", axes="L", **kwargs):
        super().__init__(axes=axes, **kwargs)
        self.from_laterality = from_laterality

    def apply_transform(self, subject):
        """
        Args:
            - subject: tio.Subject, with a "laterality" field.
        Returns:
            The transformed subject
        """
        transformed = subject
        if subject["laterality"] == self.from_laterality:
            self.axes = _ensure_axes_indices(subject, self.axes)
            transformed = tio.Flip.apply_transform(self, subject)
        return transformed

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