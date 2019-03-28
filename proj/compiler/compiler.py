import sys
import numpy
import json
import string
import hashlib
import inspect
import copy
import pickle
import os
import importlib
import subprocess
import traceback
import util
import time_memory_limiter

assert sys.version_info[0] == 3, 'requires Python 3'

int_types = (int, numpy.int8, numpy.int16, numpy.int32, numpy.int64)
float_types = (float, numpy.float32, numpy.float64)

import_declare = """
import tensorflow as tf
import numpy
import json
import os
import sys; sys.path += ['../compiler']
import util
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""

main_call = """
if __name__ == "__main__":
    main()
"""

compile_filename_prefix = '_compile'

DEFAULT_VERBOSE = 0                         # Verbosity level, 0=none, 1=some, 2=more.
TENSOR_TYPE = 'tensor'
DEFAULT_DTYPE = 'tf.float32'

MODE_VARNAME = 'varname'                # Generate variable name for use in expression
MODE_SIDE_EFFECTS = 'side_effects'      # Generate ordinary statements with side effects (e.g. for loops)

DERIV_PREFIX = 'dans_d'                                 # Prefix before derivative variables
OUTPUT_ARRAY = '_output_array'
OUTPUT_PLACEHOLDER_PREFIX = '_out_'
APPROXNODE_NAME = '_approxnode'
APPROXNODE_PLACEHOLDER = 'approxnode_placeholder'

clean = True
clean_on_fail = False
allow_clean = True

use_time_memory_limiter = False

class CompilerParams:
    def __init__(self, **kw):
        self.var_list = []                  # Variables indexed by evaluation order
        self.name_to_order = {}             # Map variable name to evaluation order.
        #self.instance_dtype = {}
        self.instance_info = {}
        
        self.mode = MODE_VARNAME            # One of MODE_*.
        self.forward = True                 # If we are doing auto-diff, are we in the forard pass? (if False, in reverse pass)
        self.statement_ids = set()          # Ids of nodes that correspond to generated statements (to prevent duplicating statements)
        
        self.verbose = DEFAULT_VERBOSE      # Verbosity level, 0=none, 1=some, 2=more.
        self.check_save = False
        self.output_dtype = DEFAULT_DTYPE
        self.sanity_check = True            # If true, output code only does sanity check to see if non-approx output generated in tensorflow is the same as given output
        self.allow_g = False      # If true, use generated non-approx tensorflow code to compute output given input
        
        for (key, value) in kw.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError('unknown key for CompilerParams: %s (no corresponding attribute)'%key)

    def reset(self):
        self.statement_ids = set()
        
    def as_mode(self, mode):
        """
        A copy of self that has mode set to the given target.
        """
        ans = copy.copy(self)
        ans.mode = mode
        return ans
                
    def get_varname(self, short_name, full_name):
        """
        Convert a variable or expression name into a variable numbered by evaluation order.
        
        Here short_name is a descriptive identifier string that will be used in the variable, and
        full_name is a full unique identifier string.
        
        Return the converted variable name.
        """
        if full_name in self.name_to_order:
            return self.var_list[self.name_to_order[full_name]]
        n = len(self.var_list)
        self.name_to_order[full_name] = n
        allowed_chars = set(string.ascii_letters + string.digits)
        remain = '_' + ''.join([c if c in allowed_chars else '_' for c in short_name])
        if remain == '_':
            remain = ''
        converted_name = 'var%03d'%n + remain
        self.var_list.append(converted_name)
        return converted_name
        
    def get_deriv_name(self, short_name, full_name):
        ans = self.get_varname(short_name, full_name)[3:]
        return DERIV_PREFIX + ans
        
    def var_declarations(self):
        placeholder_declare = [name + " = tf.placeholder(" + value['dtype'] + ", shape=" + str(value['shape']) + ")" for (name, value) in self.instance_info.items()]
        add_names = [name + "." + util.COMPILE_NAME_ATTR + " = '" + name + "'" for name in self.instance_info.keys()]
        if self.allow_g:
            assert len(self.instance_info) == 1
            placeholder_declare += [OUTPUT_PLACEHOLDER_PREFIX + name + " = tf.placeholder(" + value['dtype'] + ", shape=" + str(value['shape']) + ")" for (name, value) in self.instance_info.items()] 
        return '\n'.join(placeholder_declare + add_names)
      
    def var_returns(self):
        return '[' + ','.join(name for name in self.instance_info.keys()) + '], ' + OUTPUT_ARRAY
      
    def approx_declaration(self, input_name=None):
        assert self.allow_g is True
        if input_name is None:
            input_name = [key for key in self.instance_info.keys()][0]
        ans = '\n' + APPROXNODE_NAME + ' = util.get_approxnode(' + ','.join([input_name, OUTPUT_PLACEHOLDER_PREFIX+input_name, OUTPUT_ARRAY]) + ')'
        return ans
      
def to_source_nonfinal(e, compiler_params):
    """
    Convert Expr to non-finalized source code (no wrapping method).
    """
    s = e.to_source(compiler_params)
    return s
        
def to_source(e, compiler_params):
    """
    Convert Expr to finalized source code, including variable and function declarations.
    """
    e.calc_parents()
        
    compiler_params.mode = MODE_SIDE_EFFECTS
    compiler_params.reset()
    
    f = to_source_nonfinal(e, compiler_params)
    f_return = '\n' + 'return tf.cast(' + e.to_source(compiler_params.as_mode(MODE_VARNAME)) + ', ' + compiler_params.output_dtype + ')'

    argument_inputs = ', '.join(compiler_params.instance_info.keys())
    f_declare = '\ndef f(' + argument_inputs + '):\n'
    f = f_declare + indent(f + f_return)
    
    input_declares = compiler_params.var_declarations()
    get_output = '\n' + OUTPUT_ARRAY + ' = f(' + argument_inputs + ')'
    
    if compiler_params.allow_g:
        get_output += compiler_params.approx_declaration()
    
    global_declares = '\n' + input_declares + get_output

    declare_sess = '\nsess = tf.Session()'
    apply_check = '\nutil.check_output(test_cases, ' + compiler_params.var_returns() + ', sess, check_save=' + str(compiler_params.check_save) + ')'
    check_return = '\nreturn True'
    #sanity_check = '\ndef sanity_check(test_cases=None):\n' + indent(input_declares + get_output + declare_sess + apply_check + check_return)
    sanity_check = '\ndef sanity_check(test_cases=None):\n' + indent(declare_sess + apply_check + check_return)
    
    ans = import_declare + f + sanity_check + global_declares# + main_call
    
    return ans

def get_module_prefix(f, c):
    src = to_source(f, c)
    return get_module(src)
    
def check(f, c, test_cases=None, extra_checks={}):
    src = to_source(f, c)
    if c.verbose > 0:
        print("generated output code")
        print(src)
        print()
    (prefix, m) = get_module(src)
    check_funcs = {}
    if c.sanity_check:
        #check_funcs.append(m.sanity_check)
        check_funcs[m.sanity_check] = (test_cases,)
    
    for key, value in extra_checks.items():
        if APPROXNODE_PLACEHOLDER in value:
            value = list(value)
            value[value.index(APPROXNODE_PLACEHOLDER)] = getattr(m, APPROXNODE_NAME)
            extra_checks[key] = tuple(value)
    check_funcs.update(extra_checks)
    
    for check_func, check_args in check_funcs.items():
        if use_time_memory_limiter:
            success = time_memory_limiter.TimeMemoryLimiter().run(check_func, check_args)
            #success = time_memory_limiter.TimeMemoryLimiter().run(check_func, (test_cases,))
        else:
            #success = check_func(test_cases)
            success = check_func(*check_args)
    
    #if ((success and clean) or ((not success) and clean_on_fail)) and allow_clean:
    if clean:
        remove_if_exists(prefix + '.py')    
    return success
        
def to_expr(const_or_expr):
    """
    Convert constant or expression typically to Expr type but with special cases for handling None or ConstExpr.
    """
    if isinstance(const_or_expr, int_types + float_types + (str,)):
        return ConstExpr(const_or_expr)
    elif isinstance(const_or_expr, type(None)):
        return None
    elif isinstance(const_or_expr, Expr):
        return const_or_expr
    raise ValueError('unknown case: ', const_or_expr)
    
def indent(s, count=4):
    lines = s.split('\n')
    return '\n'.join(' '*count + line for line in lines)
    
def identical(a, b):
    """
    Check whether either Expr or list of Exprs a and b are identical (cast elements to Expr using to_expr() if needed).
    """
    if isinstance(a, Expr):
        return a.identical(b)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            av = a[i]
            bv = b[i]
            if not isinstance(av, Expr):
                av = to_expr(av)
            if not isinstance(bv, Expr):
                bv = to_expr(bv)
            if not av.identical(bv):
                return False
        return True
        
def linenos_from_frame(current_frame, depth=10):
    ans = []
    ans.append(current_frame.f_lineno)
    for i in range(depth):
        current_frame = current_frame.f_back
        if current_frame is None:
            break
        ans.append(current_frame.f_lineno)
    return ans

class Expr:
    """
    Expression type.
    
    Attributes:
    children --- A list of children for the expression (of any type, but AST nodes are of type Expr).
    """
    def __init__(self, frame_depth=2):
        self.children = []
        current_frame = inspect.currentframe()
        self.frame_lineno = linenos_from_frame(current_frame)
        self.nchannels = -1
    
    def __copy__(self):
        """
        Create a shallow copy which does not share the children list.
        """
        cls = self.__class__
        ans = cls.__new__(cls)
        ans.__dict__.update(self.__dict__)
        ans.children = list(ans.children)
        return ans
        
    def unique_str(self):
        ans = self.repr(False)
        return ans
        
    def repr(self, extra_info=True, cache=None):
        if cache is None:
            cache = dict()
        cache_key = id(self)
        if cache_key in cache:
            extra_s = ''
            if extra_info:
                extra_s = 'id: ' + str(id(self)) + ', '
            return self.__class__.__name__ + '(%srepeated %s)' % (extra_s, cache[cache_key])
            
        line_length = 80
        sub_repr = [x.repr(extra_info, cache) if hasattr(x, 'repr') else repr(x) for x in self.children]
        sub_repr_len = sum([len(x) for x in sub_repr])
        
        if extra_info:
            if hasattr(self, 'parents'):        # If calc_parents() called, include extra info in repr().
                sub_repr = ['parents: [' + ', '.join(str(id(p)) for p in self.parents) + ']'] + sub_repr
            sub_repr = ['id: ' + str(id(self))] + sub_repr
            
        if sub_repr_len < line_length and all([not hasattr(node, 'children') or node.children == [] or isinstance(node, ConstExpr) for node in self.children]):
            ans = self.__class__.__name__ + '(' + (', '.join(sub_repr)) + ')'
        else:
            ans = self.__class__.__name__ + '(\n' + indent(',\n'.join(sub_repr)) + '\n)'
            
        cache[cache_key] = hashlib.md5(ans.encode('utf-8')).hexdigest()
        return ans
        
        
    def identical(self, b):
        """
        Returns bool for whether self and b are identical expressions without attempting any simplification.
        """
        return self.unique_str() == b.unique_str()
        
    def simplify(self, seen=None):
        """
        Simplifies the given Expr in place (recursively), returning the simplified Expr.
        """
        if seen is None:
            seen = set()
        id_self = id(self)
        if id_self in seen:
            return self
        seen.add(id_self)
        self.simplify_children(seen)
        return self.simplify_impl()
        
    def simplify_impl(self):
        """
        Simplifies the given Expr but not its children in place, returning the simplified Expr.
        """
        return self
        
    def simplify_children(self, seen):
        """
        Simplifies the children of the given Expr in place.
        """
        for (i, child) in enumerate(self.children):
            if hasattr(child, 'simplify'):
                if id(child) not in seen:
                    cp = child.simplify(seen)
                    if not isinstance(cp, Expr) and cp is not None:
                        raise ValueError('Bad type for simplified expression', (cp, type(cp), child))
                    self.children[i] = cp
                    
    def clear_parents(self):
        for node in self.all_nodes():
            if hasattr(node, 'parents'):
                del node.parents
                
    def calc_parents(self):
        for node in self.all_nodes():
            node.parents = []
        for node in self.all_nodes():
            for child in node.children:
                if isinstance(child, Expr):
                    child.parents.append(node)
                    
    def is_constant(self):
        """
        Is this a constant expression? (ConstExpr or a tree containing operators, calls, and constants)
        """
        return all(isinstance(node, (ConstExpr, BinaryOp, UnaryOp, Call)) for node in self.all_nodes())
        
    def check_acyclic(self, allow_visit=None):
        """
        Raise an exception if there is a cycle in the Expr graph. Otherwise, return self.
        """
        seen = set()
        ans = []
        def visit(node, parents):
            if id(node) in parents:
                raise ValueError('cycle detected')
            if id(node) in seen:
                return
            seen.add(id(node))
            parents = parents | set([id(node)])
            for child in node.children:
                if isinstance(child, Expr):
                    if (allow_visit is None or allow_visit(node, child)):
                        visit(child, parents)
            ans.append(node)
        visit(self, set())
        return self
        
    def all_nodes(self, allow_visit=None):
        """
        Recursive nodes including self (Expr subclasses only), in breadth-first order, top down.
        
        If allow_visit is not None then allow_visit(parent, child) should give a bool for whether to visit the given child.
        """
        ans = [self]
        seen = set([id(self)])
        def visit(node):
            visit_next = []
            for child in node.children:
                if isinstance(child, Expr):
                    if id(child) not in seen and (allow_visit is None or allow_visit(node, child)):
                        ans.append(child)
                        seen.add(id(child))
                        visit_next.append(child)
            for child in visit_next:
                visit(child)
        visit(self)
        return ans
        
    def either_name(self, compiler_params):
        """
        Get either variable name (if forward pass) or derivative name (if reverse pass).
        """
        return self.deriv_name(compiler_params) if not compiler_params.forward else self.var_name(compiler_params)
        
    def var_name(self, compiler_params):
        ans = compiler_params.get_varname(getattr(self, 'name', ''), id(self))
        if compiler_params.verbose >= 2:
            print('var_name:', repr(self), compiler_params, ans)
        return ans
        
    def deriv_name(self, compiler_params):
        ans = compiler_params.get_deriv_name(getattr(self, 'name', ''), id(self))
        return 'invalid_' + ans
        
    def to_source_expr(self, compiler_params):
        """
        Convert a side-effect free expression to source code.
        """
        raise NotImplementedError(self.__class__)
        
    def debug_return(self, compiler_params, retval):
        if retval is None:
            if compiler_params.verbose >= 2:
                print('repr(self):', repr(self))
            raise ValueError
        if compiler_params.verbose >= 2:
            print(self.__class__.__name__ + '.to_source (ret) ', compiler_params, '|', retval, '|')
        return retval
        
    def is_inline(self, compiler_params):
        return False
    
    def to_source_inline(self, compiler_params):
        if compiler_params.mode == MODE_VARNAME:
            return self.to_source_impl(compiler_params)
        elif compiler_params.mode == MODE_SIDE_EFFECTS:
            return ''
        else:
            raise ValueError('unknown mode', compiler_params.mode)
        
    def to_source(self, compiler_params):
        if self.is_inline(compiler_params):
            ans = self.to_source_inline(compiler_params)
        else:
            ans = self.to_source_impl(compiler_params)
            ans = self.debug_return(compiler_params, ans)
        return ans
        
    def to_source_recurse(self, compiler_params, ans):
        cp = compiler_params.as_mode(MODE_SIDE_EFFECTS)
        children = [child for child in self.children if isinstance(child, Expr)]
        prepend = ''
        for child in children:
            sub = child.to_source(cp)
            if len(sub):
                prepend = prepend + '\n' + sub
        return prepend + '\n' + ans
        
    def statement_id(self, compiler_params):
        """
        Get an "id" key for the set CompilerParams.statement_ids.
        """
        return (compiler_params.forward, id(self))
        
    def to_source_impl(self, compiler_params):
        """
        Convert to source code in the given language.
        
        In the base class this assumes a side-effect free expression is used, and implemented in to_source_expr().
        However, this behavior can be overridden by implementing a different to_source() in subclasses.
        """
        if compiler_params.mode == MODE_VARNAME:
            return self.either_name(compiler_params)
        elif compiler_params.mode == MODE_SIDE_EFFECTS:
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                return ''
            compiler_params.statement_ids.add(self_id)
            rhs = self.to_source_expr(compiler_params.as_mode(MODE_VARNAME))
            if compiler_params.verbose > 1:
                print('to_source_impl to_source_rhs, id:', id(self), 'statement_id:', repr(self_id))
                print('to_source_impl returns:', id(self), rhs)
                
            ans = self.to_source_recurse(compiler_params, '')
            
            def ljust(s):
                return (s).ljust(50)
            
            rhs_comment = ' # Expr, id: ' + str(id(self)) + ', Linenos for Expr: ' + str(self.frame_lineno) + ', Linenos for codegen: ' + str(linenos_from_frame(inspect.currentframe()))
            ans += ljust(self.either_name(compiler_params) + ' = ' + rhs) + rhs_comment
            return ans
        else:
            raise ValueError('unhandled mode', compiler_params.mode)
            
    def __repr__(self):
        return self.repr()
    
    def __str__(self):
        return self.__class__.__name__
        
    def __add__(self, other):
        return BinaryOp('+', self, other)
        
    def __radd__(self, other):
        return BinaryOp('+', other, self)
        
    def __mul__(self, other):
        return BinaryOp('*', self, other)
        
    def __rmul__(self, other):
        return BinaryOp('*', other, self)
        
    def __truediv__(self, other):
        return BinaryOp('/', self, other)
        
    def __rtruediv__(self, other):
        return BinaryOp('/', other, self)
        
    def __sub__(self, other):
        return BinaryOp('-', self, other)
    
    def __rsub__(self, other):
        return BinaryOp('-', other, self)
        
    def __pow__(self, other):
        if isinstance(other, int_types + float_types):
            if other == 1:
                return self
            elif other == 0:
                return 1.0
        return BinaryOp('**', self, other)
        
    def __rpow__(self, other):
        return BinaryOp('**', other, self)
        
    def __lt__(self, other):
        return BinaryOp('<', self, other)
    
    def __le__(self, other):
        return BinaryOp('<=', self, other)
        
    def __eq__(self, other):
        return BinaryOp('==', self, other)
        
    def __ne__(self, other):
        return BinaryOp('!=', self, other)
        
    def __gt__(self, other):
        return BinaryOp('>', self, other)
        
    def __ge__(self, other):
        return BinaryOp('>=', self, other)
        
class AlwaysInlineExpr(Expr):
    """
    Expression that is always generated inline. Subclasses should implement to_source_impl() method.
    """
    def to_source_impl(self, compiler_params):
        raise NotImplementedError
    
    def is_inline(self, compiler_params):
        return True
        
class ConstExpr(AlwaysInlineExpr):
    def __init__(self, value):
        super().__init__()
        self.value = value
        
    def repr(self, extra_info=True, cache=None):
        return str(self.value)
    
    def __str__(self):
        return super().__str__() + '(' + str(self.value) + ')'
    
    def to_source_impl(self, compiler_params):
        return repr(self)
        
class ConstArrayExpr(Expr):
    """
    Convert such array into tensors
    """
    def __init__(self, value, dtype=DEFAULT_DTYPE):
        super().__init__()
        self.value = value
        self.dtype = dtype
        assert isinstance(value, numpy.ndarray)
        self.shape = value.shape
        
    def to_source_expr(self, compiler_params):
        #value_str = str(self.value.tostring())
        #shape_str = str(self.shape)
        value_str = json.dumps(numpy.ndarray.tolist(self.value))
        return "tf.constant(numpy.asarray(json.loads('" + value_str + "')), dtype=" + self.dtype + ")"
        
def gen_attrs(assign_lambdas):
    ans = []
    for (i, assign_lambda) in enumerate(assign_lambdas):
        def append_setter(i, assign_lambda):
            def setter(self, value):
                self.children[i] = assign_lambda(value)
            ans.append(property(lambda self: self.children[i], setter))
        append_setter(i, assign_lambda)
    return ans
        
class Var(Expr):
    """
    A variable that is assigned initially.
    """
    def __init__(self, name, initial_value=0.0):
        super().__init__()
        self.children = [str(name), to_expr(initial_value)]
        
    (name, initial_value, reduce, reduce_value) = gen_attrs([str, to_expr, str, to_expr])
    
class ArgumentArray(AlwaysInlineExpr):
    """
    ArgumentArray are always assumed to be arrays, not scalars
    and will be translated into tensors in the output code
    """
    def __init__(self, name, dtype=DEFAULT_DTYPE, shape=None):
        super().__init__()
        self.dtype = dtype
        self.shape = shape
        if self.shape is not None:
            if isinstance(self.shape, (list, tuple)):
                self.ndims = len(self.shape)
            else:
                self.ndims = 1
        else:
            self.ndims = None
        self.children = [str(name)]
        
    (name,) = gen_attrs([str])
    
    def to_source_impl(self, compiler_params):
        #compiler_params.instance_dtype[self.name] = self.dtype
        compiler_params.instance_info[self.name] = {'dtype': self.dtype, 'shape': self.shape}
        return self.name
            
class BinaryOp(Expr):
    def __init__(self, op, a, b):
        super().__init__()
        self.children = [str(op), to_expr(a), to_expr(b)]
        assert isinstance(self.a, Expr)
        assert isinstance(self.b, Expr)
        
    (op, a, b) = gen_attrs([str, to_expr, to_expr])
    
    def __str__(self):
        return super().__str__() + '(' + self.op + ')'
        
    def to_source_expr(self, compiler_params):
        a = self.a.to_source(compiler_params)
        b = self.b.to_source(compiler_params)
        return '((' + a + ')' + self.op + '(' + b + '))'
        
class Func(Expr):
    def __init__(self, name):
        super().__init__()
        self.children = [str(name)]
        
    (name,) = gen_attrs([to_expr])
    
    def __call__(self, *args):
        return Call(self.name, *args)
    
    def to_source_expr(self, compiler_params):
        raise ValueError('Func should be called before conversion to source')
        
class Call(Expr):
    def __init__(self, name, *args):
        super().__init__()
        self.children = [str(name)] + [to_expr(a) for a in args]
        
    (name,) = gen_attrs([to_expr])
        
    def to_source_expr(self, compiler_params):
        return self.name + '(' + ','.join([a.to_source(compiler_params) for a in self.children[1:]]) + ')'
        
def get_module(source):
    prefix = compile_filename_prefix + hashlib.md5(pickle.dumps(source)).hexdigest()
    with open(prefix + '.py', 'wt') as f:
        f.write(source)
    success = True
    m = importlib.import_module(prefix)
    return (prefix, m)
            
def remove_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
class ImageGray(ArgumentArray):
    def __init__(self, name, dtype=DEFAULT_DTYPE):
        super().__init__(name, dtype, (None, None, None, 1))
        self.nchannels = 1
        
class ImageRGB(ArgumentArray):
    def __init__(self, name, dtype=DEFAULT_DTYPE):
        super().__init__(name, dtype, (None, None, None, 3))
        self.nchannels = 3
        
def conv2d(X, Y):
    """
    X should be Expr representing a 4D tensor
    Y should be a 2D conv kernel
    nchannels is the number of channels the kernle should apply to
    for example, for grayscale image, nchannels=1, for RGB image, nchannels=3
    if nchannels=-1, try to infer it from X
    """
    if X.nchannels < 0:
        raise "Unable to infer nchannels, please specify it"
        #if isinstance(X, ImageGray):
        #    nchannels = 1
        #elif isinstance(X, ImageRGB):
        #    nchannels = 3
        #else:
        #    raise "Unable to infer nchannels, please specify it"
    else:
        nchannels = X.nchannels
        
    if isinstance(Y, (numpy.ndarray, ConstArrayExpr)):
        assert len(Y.shape) == 2
        assert isinstance(Y, (numpy.ndarray, ConstArrayExpr))
        Y_value = Y if isinstance(Y, numpy.ndarray) else Y.value
        Y_dtype = Y.dtype if isinstance(Y, ConstArrayExpr) else DEFAULT_DTYPE
        
        Y_extended = numpy.zeros(Y_value.shape + (nchannels, nchannels))
        for i in range(nchannels):
            Y_extended[:, :, i, i] = Y_value[:, :]
        Y_final = ConstArrayExpr(Y_extended, dtype=Y_dtype)
    else:
        if nchannels > 1:
            raise "Unable to process this case for now"
        Y1 = tensor_expand_dims(Y, 'dim=2')
        Y_final = tensor_expand_dims(Y1, 'dim=3')
    
    return tensor_conv2d(X, Y_final)
    
array_transpose = Func('tf.transpose')
tensor_conv2d = Func('util.tensor_conv2d')
tensor_expand_dims = Func('tf.expand_dims')
rgb2gray = Func('tf.image.rgb_to_grayscale')