import copy

class Structure:
    """
    Class representing a general tree-like structure. Supports
    - look-up of children by name using dict-style indexing
    - recursive traversal
    - conversion to "pretty" string

    Parameters
    ----------
    parent : Structure or None, optional
        default is None
    name : str, optional
        default is an empty string
    content : list of hashable objects
        default is None, which maps to a list containing the empty string

    Attributes
    ----------
    name : string
    children : list of Structure
    parent : Structure or None
    content : list of hashable objects
    """

    def __init__(self, parent = None, name = '', content = None):
        if content is None:
            #content = ['']
            content = []
        self.name = name
        self.children = []
        self.parent = parent
        self.content = content

    def __getitem__(self, key):
        for child in self.children:
            if child.name == key:
                yield child

    def copy(self):
        return copy.deepcopy(self)

    def addChild(self, cls = None, name = '', content = None, position = None):
        """
        Adds a new child and returns it.

        Parameters
        ----------
        cls : class or None, optional
            class of the created child; if None, maps to self's class
        name : str, optional
            gets passed to the Structure constructor
        content : list of hashable objects
            gets passed to the Structure constructor
        position : int or None
            if specified, gives the position in the children list at which
            the new child should be inserted; defaults to None
        """
        if cls is None:
            cls = self.__class__
        child = cls(parent = self, name = name, content = content)
        if position is not None:
            self.children = (
                self.children[:position] +
                [child] +
                self.children[position:])
        else:
            self.children.append(child)
        return child

    def addAsChild(self, obj):
        """
        Adds given object `obj` as child. Takes care of assigning the
        parent attribute of `obj`.

        Parameters
        ----------
        obj : Structure
        """
        obj.parent = self
        self.children.append(obj)

    def walk(self, returnDepth = False, check = None):
        """
        Returns a generator of all nodes. If `returnDepth` is True, the
        generator yields tuples (`node`, `depth`). If `check` is specified,
        it gets called on each node to see if the walk should continue to its
        children.

        Parameters
        ----------
        returnDepth : bool, optional
            default is False
        check : callable or None
            default is None
        """
        if returnDepth:
            yield (self, 0)
            if check is None or check(self):
                for child in self.children:
                    for obj, depth in child.walk(returnDepth = True, check = check):
                        yield (obj, depth + 1)

        else:
            yield self
            if check is None or check(self):
                for child in self.children:
                    yield from child.walk(check = check)

    def __str__(self, indent = ''):
        s = indent + str(self.name) + ': ' + str(self.content) + '\n'
        for child in self.children:
            s += child.__str__(indent = '  ' + indent)
        return s

    def __hash__(self):
        return hash((self.name,) + tuple(self.content) +
            tuple([hash(child) for child in self.children]))

    def countChildren(self, name):
        count = 0
        for child in self.children:
            if child.name == name:
                count += 1
        return count

    def merge(self, other, names_toMerge = None, names_toCompare = None):
        if names_toMerge is None: names_toMerge = []
        if names_toCompare is None: names_toCompare = []
        for child in other.children:
            if child.name in names_toMerge:
                self.children.append(child)
            elif child.name in names_toCompare:
                for child2 in self.children:
                    if child2.name == child.name:
                        if hash(child) != hash(child2):
                            print('WARNING: differing entries')
                            print('--------')
                            print(str(child))
                            print('--------')
                            print(str(child2))
                            print('========')
            else:
                raise NotImplementedError(f'Element name "{child.name}" not handled.')
