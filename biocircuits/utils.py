class AttributeContainer(object):
    """Generic class to hold attributes."""
    def __init__(self, **kw):
        self.__dict__ = kw

