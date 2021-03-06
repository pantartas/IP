??'
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8??$
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
batch_normalization_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_31/gamma
?
0batch_normalization_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_31/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_31/beta
?
/batch_normalization_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_31/beta*
_output_shapes
:*
dtype0
?
conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_60/kernel
}
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*&
_output_shapes
:*
dtype0
t
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_60/bias
m
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes
:*
dtype0
?
conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_61/kernel
}
$conv2d_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_61/kernel*&
_output_shapes
:*
dtype0
t
conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_61/bias
m
"conv2d_61/bias/Read/ReadVariableOpReadVariableOpconv2d_61/bias*
_output_shapes
:*
dtype0
?
batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_32/gamma
?
0batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_32/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_32/beta
?
/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_32/beta*
_output_shapes
:*
dtype0
?
conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_62/kernel
}
$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*&
_output_shapes
:*
dtype0
t
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_62/bias
m
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes
:*
dtype0
?
conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_63/kernel
}
$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*&
_output_shapes
:*
dtype0
t
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_63/bias
m
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_57/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_57/kernel
?
.conv2d_transpose_57/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_57/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_57/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_57/bias
?
,conv2d_transpose_57/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_57/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_58/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_58/kernel
?
.conv2d_transpose_58/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_58/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_58/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_58/bias
?
,conv2d_transpose_58/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_58/bias*
_output_shapes
:*
dtype0
?
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_33/gamma
?
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_33/beta
?
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
:*
dtype0
?
conv2d_transpose_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_59/kernel
?
.conv2d_transpose_59/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_59/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_59/bias
?
,conv2d_transpose_59/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_59/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_60/kernel
?
.conv2d_transpose_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_60/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_60/bias
?
,conv2d_transpose_60/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_60/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_61/kernel
?
.conv2d_transpose_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_61/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_61/bias
?
,conv2d_transpose_61/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_61/bias*
_output_shapes
:*
dtype0
?
"batch_normalization_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_31/moving_mean
?
6batch_normalization_31/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_31/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_31/moving_variance
?
:batch_normalization_31/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_31/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_32/moving_mean
?
6batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_32/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_32/moving_variance
?
:batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_32/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_33/moving_mean
?
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_33/moving_variance
?
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_31/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_31/gamma/m
?
7Adam/batch_normalization_31/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_31/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_31/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_31/beta/m
?
6Adam/batch_normalization_31/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_31/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_60/kernel/m
?
+Adam/conv2d_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_60/bias/m
{
)Adam/conv2d_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_61/kernel/m
?
+Adam/conv2d_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_61/bias/m
{
)Adam/conv2d_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_32/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_32/gamma/m
?
7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_32/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_32/beta/m
?
6Adam/batch_normalization_32/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_62/kernel/m
?
+Adam/conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_62/bias/m
{
)Adam/conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_63/kernel/m
?
+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/m
{
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_57/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_57/kernel/m
?
5Adam/conv2d_transpose_57/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_57/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_57/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_57/bias/m
?
3Adam/conv2d_transpose_57/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_57/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_58/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_58/kernel/m
?
5Adam/conv2d_transpose_58/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_58/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_58/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_58/bias/m
?
3Adam/conv2d_transpose_58/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_58/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_33/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_33/gamma/m
?
7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_33/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_33/beta/m
?
6Adam/batch_normalization_33/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_59/kernel/m
?
5Adam/conv2d_transpose_59/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_59/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_59/bias/m
?
3Adam/conv2d_transpose_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_59/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_60/kernel/m
?
5Adam/conv2d_transpose_60/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_60/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_60/bias/m
?
3Adam/conv2d_transpose_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_60/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_61/kernel/m
?
5Adam/conv2d_transpose_61/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_61/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_61/bias/m
?
3Adam/conv2d_transpose_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_61/bias/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_31/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_31/gamma/v
?
7Adam/batch_normalization_31/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_31/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_31/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_31/beta/v
?
6Adam/batch_normalization_31/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_31/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_60/kernel/v
?
+Adam/conv2d_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_60/bias/v
{
)Adam/conv2d_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_61/kernel/v
?
+Adam/conv2d_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_61/bias/v
{
)Adam/conv2d_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_32/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_32/gamma/v
?
7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_32/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_32/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_32/beta/v
?
6Adam/batch_normalization_32/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_32/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_62/kernel/v
?
+Adam/conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_62/bias/v
{
)Adam/conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_63/kernel/v
?
+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_63/bias/v
{
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_57/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_57/kernel/v
?
5Adam/conv2d_transpose_57/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_57/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_57/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_57/bias/v
?
3Adam/conv2d_transpose_57/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_57/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_58/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_58/kernel/v
?
5Adam/conv2d_transpose_58/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_58/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_58/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_58/bias/v
?
3Adam/conv2d_transpose_58/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_58/bias/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_33/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_33/gamma/v
?
7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/v*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_33/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_33/beta/v
?
6Adam/batch_normalization_33/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_59/kernel/v
?
5Adam/conv2d_transpose_59/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_59/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_59/bias/v
?
3Adam/conv2d_transpose_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_59/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_60/kernel/v
?
5Adam/conv2d_transpose_60/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_60/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_60/bias/v
?
3Adam/conv2d_transpose_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_60/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_61/kernel/v
?
5Adam/conv2d_transpose_61/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_61/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_61/bias/v
?
3Adam/conv2d_transpose_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_61/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
 
?

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
regularization_losses
trainable_variables
	variables
	keras_api
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
regularization_losses
trainable_variables
 	variables
!	keras_api
?
"iter

#beta_1

$beta_2
	%decay
&learning_rate'm?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v?
 
?
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23
?
'0
(1
?2
@3
)4
*5
+6
,7
-8
.9
A10
B11
/12
013
114
215
316
417
518
619
720
821
C22
D23
924
:25
;26
<27
=28
>29
?
regularization_losses

Elayers
Flayer_metrics
trainable_variables
	variables
Gnon_trainable_variables
Hlayer_regularization_losses
Imetrics
 
 
?
Jaxis
	'gamma
(beta
?moving_mean
@moving_variance
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
h

)kernel
*bias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
h

+kernel
,bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
?
Waxis
	-gamma
.beta
Amoving_mean
Bmoving_variance
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
h

/kernel
0bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
h

1kernel
2bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
R
dregularization_losses
etrainable_variables
f	variables
g	keras_api
 
V
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
v
'0
(1
?2
@3
)4
*5
+6
,7
-8
.9
A10
B11
/12
013
114
215
?
regularization_losses

hlayers
ilayer_metrics
trainable_variables
	variables
jnon_trainable_variables
klayer_regularization_losses
lmetrics
 
R
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
h

3kernel
4bias
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
h

5kernel
6bias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
?
yaxis
	7gamma
8beta
Cmoving_mean
Dmoving_variance
zregularization_losses
{trainable_variables
|	variables
}	keras_api
j

9kernel
:bias
~regularization_losses
trainable_variables
?	variables
?	keras_api
l

;kernel
<bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
l

=kernel
>bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
V
30
41
52
63
74
85
96
:7
;8
<9
=10
>11
f
30
41
52
63
74
85
C6
D7
98
:9
;10
<11
=12
>13
?
regularization_losses
?layers
?layer_metrics
trainable_variables
 	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_31/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_31/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_60/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_60/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_61/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_61/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_32/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_32/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_62/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_62/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_63/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_63/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_57/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_57/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_58/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_58/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEbatch_normalization_33/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization_33/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_59/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_59/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_60/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_60/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_61/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose_61/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"batch_normalization_31/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE&batch_normalization_31/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_32/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_32/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE"batch_normalization_33/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&batch_normalization_33/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
*
?0
@1
A2
B3
C4
D5
 

?0
 
 

'0
(1

'0
(1
?2
@3
?
Kregularization_losses
?layers
Ltrainable_variables
M	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

)0
*1

)0
*1
?
Oregularization_losses
?layers
Ptrainable_variables
Q	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

+0
,1

+0
,1
?
Sregularization_losses
?layers
Ttrainable_variables
U	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 
 

-0
.1

-0
.1
A2
B3
?
Xregularization_losses
?layers
Ytrainable_variables
Z	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

/0
01

/0
01
?
\regularization_losses
?layers
]trainable_variables
^	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

10
21

10
21
?
`regularization_losses
?layers
atrainable_variables
b	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 
 
 
?
dregularization_losses
?layers
etrainable_variables
f	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
8

0
1
2
3
4
5
6
7
 

?0
@1
A2
B3
 
 
 
 
 
?
mregularization_losses
?layers
ntrainable_variables
o	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

30
41

30
41
?
qregularization_losses
?layers
rtrainable_variables
s	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

50
61

50
61
?
uregularization_losses
?layers
vtrainable_variables
w	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 
 

70
81

70
81
C2
D3
?
zregularization_losses
?layers
{trainable_variables
|	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

90
:1

90
:1
?
~regularization_losses
?layers
trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

;0
<1

;0
<1
?
?regularization_losses
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
 

=0
>1

=0
>1
?
?regularization_losses
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
8
0
1
2
3
4
5
6
7
 

C0
D1
 
 
8

?total

?count
?	variables
?	keras_api
 
 

?0
@1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

C0
D1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
??
VARIABLE_VALUE#Adam/batch_normalization_31/gamma/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_31/beta/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_60/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_60/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_61/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_61/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_32/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_62/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_62/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_63/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_63/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_57/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_57/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_58/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_58/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_33/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_59/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_59/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_60/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_60/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_61/kernel/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_61/bias/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_31/gamma/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_31/beta/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_60/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_60/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_61/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_61/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_32/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_32/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_62/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_62/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_63/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_63/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_57/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_57/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_58/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_58/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_33/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_59/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_59/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_60/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_60/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/conv2d_transpose_61/kernel/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/conv2d_transpose_61/bias/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_inputPlaceholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputbatch_normalization_31/gammabatch_normalization_31/beta"batch_normalization_31/moving_mean&batch_normalization_31/moving_varianceconv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_transpose_57/kernelconv2d_transpose_57/biasconv2d_transpose_58/kernelconv2d_transpose_58/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_transpose_59/kernelconv2d_transpose_59/biasconv2d_transpose_60/kernelconv2d_transpose_60/biasconv2d_transpose_61/kernelconv2d_transpose_61/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_5680116
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp0batch_normalization_31/gamma/Read/ReadVariableOp/batch_normalization_31/beta/Read/ReadVariableOp$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp$conv2d_61/kernel/Read/ReadVariableOp"conv2d_61/bias/Read/ReadVariableOp0batch_normalization_32/gamma/Read/ReadVariableOp/batch_normalization_32/beta/Read/ReadVariableOp$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp.conv2d_transpose_57/kernel/Read/ReadVariableOp,conv2d_transpose_57/bias/Read/ReadVariableOp.conv2d_transpose_58/kernel/Read/ReadVariableOp,conv2d_transpose_58/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp.conv2d_transpose_59/kernel/Read/ReadVariableOp,conv2d_transpose_59/bias/Read/ReadVariableOp.conv2d_transpose_60/kernel/Read/ReadVariableOp,conv2d_transpose_60/bias/Read/ReadVariableOp.conv2d_transpose_61/kernel/Read/ReadVariableOp,conv2d_transpose_61/bias/Read/ReadVariableOp6batch_normalization_31/moving_mean/Read/ReadVariableOp:batch_normalization_31/moving_variance/Read/ReadVariableOp6batch_normalization_32/moving_mean/Read/ReadVariableOp:batch_normalization_32/moving_variance/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp7Adam/batch_normalization_31/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_31/beta/m/Read/ReadVariableOp+Adam/conv2d_60/kernel/m/Read/ReadVariableOp)Adam/conv2d_60/bias/m/Read/ReadVariableOp+Adam/conv2d_61/kernel/m/Read/ReadVariableOp)Adam/conv2d_61/bias/m/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_32/beta/m/Read/ReadVariableOp+Adam/conv2d_62/kernel/m/Read/ReadVariableOp)Adam/conv2d_62/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_57/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_57/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_58/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_58/bias/m/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_33/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_59/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_59/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_60/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_60/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_61/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_61/bias/m/Read/ReadVariableOp7Adam/batch_normalization_31/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_31/beta/v/Read/ReadVariableOp+Adam/conv2d_60/kernel/v/Read/ReadVariableOp)Adam/conv2d_60/bias/v/Read/ReadVariableOp+Adam/conv2d_61/kernel/v/Read/ReadVariableOp)Adam/conv2d_61/bias/v/Read/ReadVariableOp7Adam/batch_normalization_32/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_32/beta/v/Read/ReadVariableOp+Adam/conv2d_62/kernel/v/Read/ReadVariableOp)Adam/conv2d_62/bias/v/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_57/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_57/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_58/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_58/bias/v/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_33/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_59/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_59/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_60/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_60/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_61/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_61/bias/v/Read/ReadVariableOpConst*b
Tin[
Y2W	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_5682272
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratebatch_normalization_31/gammabatch_normalization_31/betaconv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasbatch_normalization_32/gammabatch_normalization_32/betaconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_transpose_57/kernelconv2d_transpose_57/biasconv2d_transpose_58/kernelconv2d_transpose_58/biasbatch_normalization_33/gammabatch_normalization_33/betaconv2d_transpose_59/kernelconv2d_transpose_59/biasconv2d_transpose_60/kernelconv2d_transpose_60/biasconv2d_transpose_61/kernelconv2d_transpose_61/bias"batch_normalization_31/moving_mean&batch_normalization_31/moving_variance"batch_normalization_32/moving_mean&batch_normalization_32/moving_variance"batch_normalization_33/moving_mean&batch_normalization_33/moving_variancetotalcount#Adam/batch_normalization_31/gamma/m"Adam/batch_normalization_31/beta/mAdam/conv2d_60/kernel/mAdam/conv2d_60/bias/mAdam/conv2d_61/kernel/mAdam/conv2d_61/bias/m#Adam/batch_normalization_32/gamma/m"Adam/batch_normalization_32/beta/mAdam/conv2d_62/kernel/mAdam/conv2d_62/bias/mAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/m!Adam/conv2d_transpose_57/kernel/mAdam/conv2d_transpose_57/bias/m!Adam/conv2d_transpose_58/kernel/mAdam/conv2d_transpose_58/bias/m#Adam/batch_normalization_33/gamma/m"Adam/batch_normalization_33/beta/m!Adam/conv2d_transpose_59/kernel/mAdam/conv2d_transpose_59/bias/m!Adam/conv2d_transpose_60/kernel/mAdam/conv2d_transpose_60/bias/m!Adam/conv2d_transpose_61/kernel/mAdam/conv2d_transpose_61/bias/m#Adam/batch_normalization_31/gamma/v"Adam/batch_normalization_31/beta/vAdam/conv2d_60/kernel/vAdam/conv2d_60/bias/vAdam/conv2d_61/kernel/vAdam/conv2d_61/bias/v#Adam/batch_normalization_32/gamma/v"Adam/batch_normalization_32/beta/vAdam/conv2d_62/kernel/vAdam/conv2d_62/bias/vAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/v!Adam/conv2d_transpose_57/kernel/vAdam/conv2d_transpose_57/bias/v!Adam/conv2d_transpose_58/kernel/vAdam/conv2d_transpose_58/bias/v#Adam/batch_normalization_33/gamma/v"Adam/batch_normalization_33/beta/v!Adam/conv2d_transpose_59/kernel/vAdam/conv2d_transpose_59/bias/v!Adam/conv2d_transpose_60/kernel/vAdam/conv2d_transpose_60/bias/v!Adam/conv2d_transpose_61/kernel/vAdam/conv2d_transpose_61/bias/v*a
TinZ
X2V*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_5682537??!
?	
?
8__inference_batch_normalization_31_layer_call_fn_5681147

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_56776392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?	
H__inference_autoencoder_layer_call_and_return_conditional_losses_5679977	
input
encoder_5679914:
encoder_5679916:
encoder_5679918:
encoder_5679920:)
encoder_5679922:
encoder_5679924:)
encoder_5679926:
encoder_5679928:
encoder_5679930:
encoder_5679932:
encoder_5679934:
encoder_5679936:)
encoder_5679938:
encoder_5679940:)
encoder_5679942:
encoder_5679944:)
decoder_5679947:
decoder_5679949:)
decoder_5679951:
decoder_5679953:
decoder_5679955:
decoder_5679957:
decoder_5679959:
decoder_5679961:)
decoder_5679963:
decoder_5679965:)
decoder_5679967:
decoder_5679969:)
decoder_5679971:
decoder_5679973:
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputencoder_5679914encoder_5679916encoder_5679918encoder_5679920encoder_5679922encoder_5679924encoder_5679926encoder_5679928encoder_5679930encoder_5679932encoder_5679934encoder_5679936encoder_5679938encoder_5679940encoder_5679942encoder_5679944*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56780072!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_5679947decoder_5679949decoder_5679951decoder_5679953decoder_5679955decoder_5679957decoder_5679959decoder_5679961decoder_5679963decoder_5679965decoder_5679967decoder_5679969decoder_5679971decoder_5679973*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56791752!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:V R
/
_output_shapes
:?????????@@

_user_specified_nameinput
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5677683

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5678614

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5681920

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_61_layer_call_fn_5681938

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_56791682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?"
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680616

inputsD
6encoder_batch_normalization_31_readvariableop_resource:F
8encoder_batch_normalization_31_readvariableop_1_resource:U
Gencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_resource:W
Iencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:J
0encoder_conv2d_60_conv2d_readvariableop_resource:?
1encoder_conv2d_60_biasadd_readvariableop_resource:J
0encoder_conv2d_61_conv2d_readvariableop_resource:?
1encoder_conv2d_61_biasadd_readvariableop_resource:D
6encoder_batch_normalization_32_readvariableop_resource:F
8encoder_batch_normalization_32_readvariableop_1_resource:U
Gencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:W
Iencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:J
0encoder_conv2d_62_conv2d_readvariableop_resource:?
1encoder_conv2d_62_biasadd_readvariableop_resource:J
0encoder_conv2d_63_conv2d_readvariableop_resource:?
1encoder_conv2d_63_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_57_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_57_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_58_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_58_biasadd_readvariableop_resource:D
6decoder_batch_normalization_33_readvariableop_resource:F
8decoder_batch_normalization_33_readvariableop_1_resource:U
Gdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:W
Idecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:^
Ddecoder_conv2d_transpose_59_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_59_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_60_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_60_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_61_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_61_biasadd_readvariableop_resource:
identity??-decoder/batch_normalization_33/AssignNewValue?/decoder/batch_normalization_33/AssignNewValue_1?>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?-decoder/batch_normalization_33/ReadVariableOp?/decoder/batch_normalization_33/ReadVariableOp_1?2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp?-encoder/batch_normalization_31/AssignNewValue?/encoder/batch_normalization_31/AssignNewValue_1?>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?-encoder/batch_normalization_31/ReadVariableOp?/encoder/batch_normalization_31/ReadVariableOp_1?-encoder/batch_normalization_32/AssignNewValue?/encoder/batch_normalization_32/AssignNewValue_1?>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?-encoder/batch_normalization_32/ReadVariableOp?/encoder/batch_normalization_32/ReadVariableOp_1?(encoder/conv2d_60/BiasAdd/ReadVariableOp?'encoder/conv2d_60/Conv2D/ReadVariableOp?(encoder/conv2d_61/BiasAdd/ReadVariableOp?'encoder/conv2d_61/Conv2D/ReadVariableOp?(encoder/conv2d_62/BiasAdd/ReadVariableOp?'encoder/conv2d_62/Conv2D/ReadVariableOp?(encoder/conv2d_63/BiasAdd/ReadVariableOp?'encoder/conv2d_63/Conv2D/ReadVariableOp?
-encoder/batch_normalization_31/ReadVariableOpReadVariableOp6encoder_batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype02/
-encoder/batch_normalization_31/ReadVariableOp?
/encoder/batch_normalization_31/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype021
/encoder/batch_normalization_31/ReadVariableOp_1?
>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
/encoder/batch_normalization_31/FusedBatchNormV3FusedBatchNormV3inputs5encoder/batch_normalization_31/ReadVariableOp:value:07encoder/batch_normalization_31/ReadVariableOp_1:value:0Fencoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0Hencoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/encoder/batch_normalization_31/FusedBatchNormV3?
-encoder/batch_normalization_31/AssignNewValueAssignVariableOpGencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_resource<encoder/batch_normalization_31/FusedBatchNormV3:batch_mean:0?^encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-encoder/batch_normalization_31/AssignNewValue?
/encoder/batch_normalization_31/AssignNewValue_1AssignVariableOpIencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource@encoder/batch_normalization_31/FusedBatchNormV3:batch_variance:0A^encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/encoder/batch_normalization_31/AssignNewValue_1?
'encoder/conv2d_60/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_60/Conv2D/ReadVariableOp?
encoder/conv2d_60/Conv2DConv2D3encoder/batch_normalization_31/FusedBatchNormV3:y:0/encoder/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
encoder/conv2d_60/Conv2D?
(encoder/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_60/BiasAdd/ReadVariableOp?
encoder/conv2d_60/BiasAddBiasAdd!encoder/conv2d_60/Conv2D:output:00encoder/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
encoder/conv2d_60/BiasAdd?
encoder/conv2d_60/ReluRelu"encoder/conv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
encoder/conv2d_60/Relu?
'encoder/conv2d_61/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_61/Conv2D/ReadVariableOp?
encoder/conv2d_61/Conv2DConv2D$encoder/conv2d_60/Relu:activations:0/encoder/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
encoder/conv2d_61/Conv2D?
(encoder/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_61/BiasAdd/ReadVariableOp?
encoder/conv2d_61/BiasAddBiasAdd!encoder/conv2d_61/Conv2D:output:00encoder/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_61/BiasAdd?
encoder/conv2d_61/ReluRelu"encoder/conv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_61/Relu?
-encoder/batch_normalization_32/ReadVariableOpReadVariableOp6encoder_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype02/
-encoder/batch_normalization_32/ReadVariableOp?
/encoder/batch_normalization_32/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype021
/encoder/batch_normalization_32/ReadVariableOp_1?
>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
/encoder/batch_normalization_32/FusedBatchNormV3FusedBatchNormV3$encoder/conv2d_61/Relu:activations:05encoder/batch_normalization_32/ReadVariableOp:value:07encoder/batch_normalization_32/ReadVariableOp_1:value:0Fencoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Hencoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/encoder/batch_normalization_32/FusedBatchNormV3?
-encoder/batch_normalization_32/AssignNewValueAssignVariableOpGencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_resource<encoder/batch_normalization_32/FusedBatchNormV3:batch_mean:0?^encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-encoder/batch_normalization_32/AssignNewValue?
/encoder/batch_normalization_32/AssignNewValue_1AssignVariableOpIencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource@encoder/batch_normalization_32/FusedBatchNormV3:batch_variance:0A^encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/encoder/batch_normalization_32/AssignNewValue_1?
'encoder/conv2d_62/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_62/Conv2D/ReadVariableOp?
encoder/conv2d_62/Conv2DConv2D3encoder/batch_normalization_32/FusedBatchNormV3:y:0/encoder/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
encoder/conv2d_62/Conv2D?
(encoder/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_62/BiasAdd/ReadVariableOp?
encoder/conv2d_62/BiasAddBiasAdd!encoder/conv2d_62/Conv2D:output:00encoder/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_62/BiasAdd?
encoder/conv2d_62/ReluRelu"encoder/conv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_62/Relu?
'encoder/conv2d_63/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_63/Conv2D/ReadVariableOp?
encoder/conv2d_63/Conv2DConv2D$encoder/conv2d_62/Relu:activations:0/encoder/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder/conv2d_63/Conv2D?
(encoder/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_63/BiasAdd/ReadVariableOp?
encoder/conv2d_63/BiasAddBiasAdd!encoder/conv2d_63/Conv2D:output:00encoder/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d_63/BiasAdd?
encoder/conv2d_63/ReluRelu"encoder/conv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d_63/Relu?
encoder/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
encoder/flatten_8/Const?
encoder/flatten_8/ReshapeReshape$encoder/conv2d_63/Relu:activations:0 encoder/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten_8/Reshape?
decoder/reshape_8/ShapeShape"encoder/flatten_8/Reshape:output:0*
T0*
_output_shapes
:2
decoder/reshape_8/Shape?
%decoder/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder/reshape_8/strided_slice/stack?
'decoder/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_8/strided_slice/stack_1?
'decoder/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_8/strided_slice/stack_2?
decoder/reshape_8/strided_sliceStridedSlice decoder/reshape_8/Shape:output:0.decoder/reshape_8/strided_slice/stack:output:00decoder/reshape_8/strided_slice/stack_1:output:00decoder/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder/reshape_8/strided_slice?
!decoder/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_8/Reshape/shape/1?
!decoder/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_8/Reshape/shape/2?
!decoder/reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_8/Reshape/shape/3?
decoder/reshape_8/Reshape/shapePack(decoder/reshape_8/strided_slice:output:0*decoder/reshape_8/Reshape/shape/1:output:0*decoder/reshape_8/Reshape/shape/2:output:0*decoder/reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/reshape_8/Reshape/shape?
decoder/reshape_8/ReshapeReshape"encoder/flatten_8/Reshape:output:0(decoder/reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
decoder/reshape_8/Reshape?
!decoder/conv2d_transpose_57/ShapeShape"decoder/reshape_8/Reshape:output:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_57/Shape?
/decoder/conv2d_transpose_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_57/strided_slice/stack?
1decoder/conv2d_transpose_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_57/strided_slice/stack_1?
1decoder/conv2d_transpose_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_57/strided_slice/stack_2?
)decoder/conv2d_transpose_57/strided_sliceStridedSlice*decoder/conv2d_transpose_57/Shape:output:08decoder/conv2d_transpose_57/strided_slice/stack:output:0:decoder/conv2d_transpose_57/strided_slice/stack_1:output:0:decoder/conv2d_transpose_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_57/strided_slice?
#decoder/conv2d_transpose_57/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_57/stack/1?
#decoder/conv2d_transpose_57/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_57/stack/2?
#decoder/conv2d_transpose_57/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_57/stack/3?
!decoder/conv2d_transpose_57/stackPack2decoder/conv2d_transpose_57/strided_slice:output:0,decoder/conv2d_transpose_57/stack/1:output:0,decoder/conv2d_transpose_57/stack/2:output:0,decoder/conv2d_transpose_57/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_57/stack?
1decoder/conv2d_transpose_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_57/strided_slice_1/stack?
3decoder/conv2d_transpose_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_57/strided_slice_1/stack_1?
3decoder/conv2d_transpose_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_57/strided_slice_1/stack_2?
+decoder/conv2d_transpose_57/strided_slice_1StridedSlice*decoder/conv2d_transpose_57/stack:output:0:decoder/conv2d_transpose_57/strided_slice_1/stack:output:0<decoder/conv2d_transpose_57/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_57/strided_slice_1?
;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_57_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_57/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_57/stack:output:0Cdecoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_8/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2.
,decoder/conv2d_transpose_57/conv2d_transpose?
2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_57/BiasAddBiasAdd5decoder/conv2d_transpose_57/conv2d_transpose:output:0:decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2%
#decoder/conv2d_transpose_57/BiasAdd?
 decoder/conv2d_transpose_57/ReluRelu,decoder/conv2d_transpose_57/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 decoder/conv2d_transpose_57/Relu?
!decoder/conv2d_transpose_58/ShapeShape.decoder/conv2d_transpose_57/Relu:activations:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_58/Shape?
/decoder/conv2d_transpose_58/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_58/strided_slice/stack?
1decoder/conv2d_transpose_58/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_58/strided_slice/stack_1?
1decoder/conv2d_transpose_58/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_58/strided_slice/stack_2?
)decoder/conv2d_transpose_58/strided_sliceStridedSlice*decoder/conv2d_transpose_58/Shape:output:08decoder/conv2d_transpose_58/strided_slice/stack:output:0:decoder/conv2d_transpose_58/strided_slice/stack_1:output:0:decoder/conv2d_transpose_58/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_58/strided_slice?
#decoder/conv2d_transpose_58/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_58/stack/1?
#decoder/conv2d_transpose_58/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_58/stack/2?
#decoder/conv2d_transpose_58/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_58/stack/3?
!decoder/conv2d_transpose_58/stackPack2decoder/conv2d_transpose_58/strided_slice:output:0,decoder/conv2d_transpose_58/stack/1:output:0,decoder/conv2d_transpose_58/stack/2:output:0,decoder/conv2d_transpose_58/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_58/stack?
1decoder/conv2d_transpose_58/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_58/strided_slice_1/stack?
3decoder/conv2d_transpose_58/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_58/strided_slice_1/stack_1?
3decoder/conv2d_transpose_58/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_58/strided_slice_1/stack_2?
+decoder/conv2d_transpose_58/strided_slice_1StridedSlice*decoder/conv2d_transpose_58/stack:output:0:decoder/conv2d_transpose_58/strided_slice_1/stack:output:0<decoder/conv2d_transpose_58/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_58/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_58/strided_slice_1?
;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_58_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_58/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_58/stack:output:0Cdecoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_57/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2.
,decoder/conv2d_transpose_58/conv2d_transpose?
2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_58/BiasAddBiasAdd5decoder/conv2d_transpose_58/conv2d_transpose:output:0:decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2%
#decoder/conv2d_transpose_58/BiasAdd?
 decoder/conv2d_transpose_58/ReluRelu,decoder/conv2d_transpose_58/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2"
 decoder/conv2d_transpose_58/Relu?
-decoder/batch_normalization_33/ReadVariableOpReadVariableOp6decoder_batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder/batch_normalization_33/ReadVariableOp?
/decoder/batch_normalization_33/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype021
/decoder/batch_normalization_33/ReadVariableOp_1?
>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
/decoder/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3.decoder/conv2d_transpose_58/Relu:activations:05decoder/batch_normalization_33/ReadVariableOp:value:07decoder/batch_normalization_33/ReadVariableOp_1:value:0Fdecoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/decoder/batch_normalization_33/FusedBatchNormV3?
-decoder/batch_normalization_33/AssignNewValueAssignVariableOpGdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_resource<decoder/batch_normalization_33/FusedBatchNormV3:batch_mean:0?^decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-decoder/batch_normalization_33/AssignNewValue?
/decoder/batch_normalization_33/AssignNewValue_1AssignVariableOpIdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource@decoder/batch_normalization_33/FusedBatchNormV3:batch_variance:0A^decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/decoder/batch_normalization_33/AssignNewValue_1?
!decoder/conv2d_transpose_59/ShapeShape3decoder/batch_normalization_33/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_59/Shape?
/decoder/conv2d_transpose_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_59/strided_slice/stack?
1decoder/conv2d_transpose_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_59/strided_slice/stack_1?
1decoder/conv2d_transpose_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_59/strided_slice/stack_2?
)decoder/conv2d_transpose_59/strided_sliceStridedSlice*decoder/conv2d_transpose_59/Shape:output:08decoder/conv2d_transpose_59/strided_slice/stack:output:0:decoder/conv2d_transpose_59/strided_slice/stack_1:output:0:decoder/conv2d_transpose_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_59/strided_slice?
#decoder/conv2d_transpose_59/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_59/stack/1?
#decoder/conv2d_transpose_59/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_59/stack/2?
#decoder/conv2d_transpose_59/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_59/stack/3?
!decoder/conv2d_transpose_59/stackPack2decoder/conv2d_transpose_59/strided_slice:output:0,decoder/conv2d_transpose_59/stack/1:output:0,decoder/conv2d_transpose_59/stack/2:output:0,decoder/conv2d_transpose_59/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_59/stack?
1decoder/conv2d_transpose_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_59/strided_slice_1/stack?
3decoder/conv2d_transpose_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_59/strided_slice_1/stack_1?
3decoder/conv2d_transpose_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_59/strided_slice_1/stack_2?
+decoder/conv2d_transpose_59/strided_slice_1StridedSlice*decoder/conv2d_transpose_59/stack:output:0:decoder/conv2d_transpose_59/strided_slice_1/stack:output:0<decoder/conv2d_transpose_59/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_59/strided_slice_1?
;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_59_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_59/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_59/stack:output:0Cdecoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp:value:03decoder/batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2.
,decoder/conv2d_transpose_59/conv2d_transpose?
2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_59/BiasAddBiasAdd5decoder/conv2d_transpose_59/conv2d_transpose:output:0:decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2%
#decoder/conv2d_transpose_59/BiasAdd?
 decoder/conv2d_transpose_59/ReluRelu,decoder/conv2d_transpose_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2"
 decoder/conv2d_transpose_59/Relu?
!decoder/conv2d_transpose_60/ShapeShape.decoder/conv2d_transpose_59/Relu:activations:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_60/Shape?
/decoder/conv2d_transpose_60/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_60/strided_slice/stack?
1decoder/conv2d_transpose_60/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_60/strided_slice/stack_1?
1decoder/conv2d_transpose_60/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_60/strided_slice/stack_2?
)decoder/conv2d_transpose_60/strided_sliceStridedSlice*decoder/conv2d_transpose_60/Shape:output:08decoder/conv2d_transpose_60/strided_slice/stack:output:0:decoder/conv2d_transpose_60/strided_slice/stack_1:output:0:decoder/conv2d_transpose_60/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_60/strided_slice?
#decoder/conv2d_transpose_60/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_60/stack/1?
#decoder/conv2d_transpose_60/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_60/stack/2?
#decoder/conv2d_transpose_60/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_60/stack/3?
!decoder/conv2d_transpose_60/stackPack2decoder/conv2d_transpose_60/strided_slice:output:0,decoder/conv2d_transpose_60/stack/1:output:0,decoder/conv2d_transpose_60/stack/2:output:0,decoder/conv2d_transpose_60/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_60/stack?
1decoder/conv2d_transpose_60/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_60/strided_slice_1/stack?
3decoder/conv2d_transpose_60/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_60/strided_slice_1/stack_1?
3decoder/conv2d_transpose_60/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_60/strided_slice_1/stack_2?
+decoder/conv2d_transpose_60/strided_slice_1StridedSlice*decoder/conv2d_transpose_60/stack:output:0:decoder/conv2d_transpose_60/strided_slice_1/stack:output:0<decoder/conv2d_transpose_60/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_60/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_60/strided_slice_1?
;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_60_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_60/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_60/stack:output:0Cdecoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_59/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2.
,decoder/conv2d_transpose_60/conv2d_transpose?
2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_60/BiasAddBiasAdd5decoder/conv2d_transpose_60/conv2d_transpose:output:0:decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2%
#decoder/conv2d_transpose_60/BiasAdd?
 decoder/conv2d_transpose_60/ReluRelu,decoder/conv2d_transpose_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2"
 decoder/conv2d_transpose_60/Relu?
!decoder/conv2d_transpose_61/ShapeShape.decoder/conv2d_transpose_60/Relu:activations:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_61/Shape?
/decoder/conv2d_transpose_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_61/strided_slice/stack?
1decoder/conv2d_transpose_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_61/strided_slice/stack_1?
1decoder/conv2d_transpose_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_61/strided_slice/stack_2?
)decoder/conv2d_transpose_61/strided_sliceStridedSlice*decoder/conv2d_transpose_61/Shape:output:08decoder/conv2d_transpose_61/strided_slice/stack:output:0:decoder/conv2d_transpose_61/strided_slice/stack_1:output:0:decoder/conv2d_transpose_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_61/strided_slice?
#decoder/conv2d_transpose_61/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_61/stack/1?
#decoder/conv2d_transpose_61/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_61/stack/2?
#decoder/conv2d_transpose_61/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_61/stack/3?
!decoder/conv2d_transpose_61/stackPack2decoder/conv2d_transpose_61/strided_slice:output:0,decoder/conv2d_transpose_61/stack/1:output:0,decoder/conv2d_transpose_61/stack/2:output:0,decoder/conv2d_transpose_61/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_61/stack?
1decoder/conv2d_transpose_61/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_61/strided_slice_1/stack?
3decoder/conv2d_transpose_61/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_61/strided_slice_1/stack_1?
3decoder/conv2d_transpose_61/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_61/strided_slice_1/stack_2?
+decoder/conv2d_transpose_61/strided_slice_1StridedSlice*decoder/conv2d_transpose_61/stack:output:0:decoder/conv2d_transpose_61/strided_slice_1/stack:output:0<decoder/conv2d_transpose_61/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_61/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_61/strided_slice_1?
;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_61_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_61/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_61/stack:output:0Cdecoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_60/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2.
,decoder/conv2d_transpose_61/conv2d_transpose?
2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_61/BiasAddBiasAdd5decoder/conv2d_transpose_61/conv2d_transpose:output:0:decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2%
#decoder/conv2d_transpose_61/BiasAdd?
IdentityIdentity,decoder/conv2d_transpose_61/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp.^decoder/batch_normalization_33/AssignNewValue0^decoder/batch_normalization_33/AssignNewValue_1?^decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_33/ReadVariableOp0^decoder/batch_normalization_33/ReadVariableOp_13^decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp.^encoder/batch_normalization_31/AssignNewValue0^encoder/batch_normalization_31/AssignNewValue_1?^encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOpA^encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1.^encoder/batch_normalization_31/ReadVariableOp0^encoder/batch_normalization_31/ReadVariableOp_1.^encoder/batch_normalization_32/AssignNewValue0^encoder/batch_normalization_32/AssignNewValue_1?^encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOpA^encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1.^encoder/batch_normalization_32/ReadVariableOp0^encoder/batch_normalization_32/ReadVariableOp_1)^encoder/conv2d_60/BiasAdd/ReadVariableOp(^encoder/conv2d_60/Conv2D/ReadVariableOp)^encoder/conv2d_61/BiasAdd/ReadVariableOp(^encoder/conv2d_61/Conv2D/ReadVariableOp)^encoder/conv2d_62/BiasAdd/ReadVariableOp(^encoder/conv2d_62/Conv2D/ReadVariableOp)^encoder/conv2d_63/BiasAdd/ReadVariableOp(^encoder/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-decoder/batch_normalization_33/AssignNewValue-decoder/batch_normalization_33/AssignNewValue2b
/decoder/batch_normalization_33/AssignNewValue_1/decoder/batch_normalization_33/AssignNewValue_12?
>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_33/ReadVariableOp-decoder/batch_normalization_33/ReadVariableOp2b
/decoder/batch_normalization_33/ReadVariableOp_1/decoder/batch_normalization_33/ReadVariableOp_12h
2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp2^
-encoder/batch_normalization_31/AssignNewValue-encoder/batch_normalization_31/AssignNewValue2b
/encoder/batch_normalization_31/AssignNewValue_1/encoder/batch_normalization_31/AssignNewValue_12?
>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp2?
@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12^
-encoder/batch_normalization_31/ReadVariableOp-encoder/batch_normalization_31/ReadVariableOp2b
/encoder/batch_normalization_31/ReadVariableOp_1/encoder/batch_normalization_31/ReadVariableOp_12^
-encoder/batch_normalization_32/AssignNewValue-encoder/batch_normalization_32/AssignNewValue2b
/encoder/batch_normalization_32/AssignNewValue_1/encoder/batch_normalization_32/AssignNewValue_12?
>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2?
@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12^
-encoder/batch_normalization_32/ReadVariableOp-encoder/batch_normalization_32/ReadVariableOp2b
/encoder/batch_normalization_32/ReadVariableOp_1/encoder/batch_normalization_32/ReadVariableOp_12T
(encoder/conv2d_60/BiasAdd/ReadVariableOp(encoder/conv2d_60/BiasAdd/ReadVariableOp2R
'encoder/conv2d_60/Conv2D/ReadVariableOp'encoder/conv2d_60/Conv2D/ReadVariableOp2T
(encoder/conv2d_61/BiasAdd/ReadVariableOp(encoder/conv2d_61/BiasAdd/ReadVariableOp2R
'encoder/conv2d_61/Conv2D/ReadVariableOp'encoder/conv2d_61/Conv2D/ReadVariableOp2T
(encoder/conv2d_62/BiasAdd/ReadVariableOp(encoder/conv2d_62/BiasAdd/ReadVariableOp2R
'encoder/conv2d_62/Conv2D/ReadVariableOp'encoder/conv2d_62/Conv2D/ReadVariableOp2T
(encoder/conv2d_63/BiasAdd/ReadVariableOp(encoder/conv2d_63/BiasAdd/ReadVariableOp2R
'encoder/conv2d_63/Conv2D/ReadVariableOp'encoder/conv2d_63/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_decoder_layer_call_fn_5679439
placeholder!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56793752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoded audio
?	
?
8__inference_batch_normalization_31_layer_call_fn_5681160

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_56776832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681732

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_5680181

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_56795872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5681994

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_62_layer_call_fn_5681431

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_56779752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_5681462

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681750

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_33_layer_call_fn_5681657

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_56786142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_5677975

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5681844

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5678165

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?+
?
D__inference_encoder_layer_call_and_return_conditional_losses_5678416
placeholder,
batch_normalization_31_5678376:,
batch_normalization_31_5678378:,
batch_normalization_31_5678380:,
batch_normalization_31_5678382:+
conv2d_60_5678385:
conv2d_60_5678387:+
conv2d_61_5678390:
conv2d_61_5678392:,
batch_normalization_32_5678395:,
batch_normalization_32_5678397:,
batch_normalization_32_5678399:,
batch_normalization_32_5678401:+
conv2d_62_5678404:
conv2d_62_5678406:+
conv2d_63_5678409:
conv2d_63_5678411:
identity??.batch_normalization_31/StatefulPartitionedCall?.batch_normalization_32/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCallplaceholderbatch_normalization_31_5678376batch_normalization_31_5678378batch_normalization_31_5678380batch_normalization_31_5678382*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_567816520
.batch_normalization_31/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_60_5678385conv2d_60_5678387*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_56779142#
!conv2d_60/StatefulPartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0conv2d_61_5678390conv2d_61_5678392*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_56779312#
!conv2d_61/StatefulPartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_32_5678395batch_normalization_32_5678397batch_normalization_32_5678399batch_normalization_32_5678401*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_567810120
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_62_5678404conv2d_62_5678406*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_56779752#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_5678409conv2d_63_5678411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_56779922#
!conv2d_63/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_56780042
flatten_8/PartitionedCall~
IdentityIdentity"flatten_8/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_31/StatefulPartitionedCall/^batch_normalization_32/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@
(
_user_specified_nameoriginal audio
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5677893

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5679111

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5681544

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5679026

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681258

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5677809

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_reshape_8_layer_call_and_return_conditional_losses_5679001

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_31_layer_call_fn_5681173

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_56778932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_5679911	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_56797832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????@@

_user_specified_nameinput
?
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_5681473

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
D__inference_encoder_layer_call_and_return_conditional_losses_5678007

inputs,
batch_normalization_31_5677894:,
batch_normalization_31_5677896:,
batch_normalization_31_5677898:,
batch_normalization_31_5677900:+
conv2d_60_5677915:
conv2d_60_5677917:+
conv2d_61_5677932:
conv2d_61_5677934:,
batch_normalization_32_5677955:,
batch_normalization_32_5677957:,
batch_normalization_32_5677959:,
batch_normalization_32_5677961:+
conv2d_62_5677976:
conv2d_62_5677978:+
conv2d_63_5677993:
conv2d_63_5677995:
identity??.batch_normalization_31/StatefulPartitionedCall?.batch_normalization_32/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_31_5677894batch_normalization_31_5677896batch_normalization_31_5677898batch_normalization_31_5677900*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_567789320
.batch_normalization_31/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_60_5677915conv2d_60_5677917*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_56779142#
!conv2d_60/StatefulPartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0conv2d_61_5677932conv2d_61_5677934*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_56779312#
!conv2d_61/StatefulPartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_32_5677955batch_normalization_32_5677957batch_normalization_32_5677959batch_normalization_32_5677961*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_567795420
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_62_5677976conv2d_62_5677978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_56779752#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_5677993conv2d_63_5677995*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_56779922#
!conv2d_63/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_56780042
flatten_8/PartitionedCall~
IdentityIdentity"flatten_8/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_31/StatefulPartitionedCall/^batch_normalization_32/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5681896

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?c
?
D__inference_encoder_layer_call_and_return_conditional_losses_5680814

inputs<
.batch_normalization_31_readvariableop_resource:>
0batch_normalization_31_readvariableop_1_resource:M
?batch_normalization_31_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_60_conv2d_readvariableop_resource:7
)conv2d_60_biasadd_readvariableop_resource:B
(conv2d_61_conv2d_readvariableop_resource:7
)conv2d_61_biasadd_readvariableop_resource:<
.batch_normalization_32_readvariableop_resource:>
0batch_normalization_32_readvariableop_1_resource:M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_62_conv2d_readvariableop_resource:7
)conv2d_62_biasadd_readvariableop_resource:B
(conv2d_63_conv2d_readvariableop_resource:7
)conv2d_63_biasadd_readvariableop_resource:
identity??%batch_normalization_31/AssignNewValue?'batch_normalization_31/AssignNewValue_1?6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_31/ReadVariableOp?'batch_normalization_31/ReadVariableOp_1?%batch_normalization_32/AssignNewValue?'batch_normalization_32/AssignNewValue_1?6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_32/ReadVariableOp?'batch_normalization_32/ReadVariableOp_1? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp? conv2d_61/BiasAdd/ReadVariableOp?conv2d_61/Conv2D/ReadVariableOp? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp?
%batch_normalization_31/ReadVariableOpReadVariableOp.batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_31/ReadVariableOp?
'batch_normalization_31/ReadVariableOp_1ReadVariableOp0batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_31/ReadVariableOp_1?
6batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_31/FusedBatchNormV3FusedBatchNormV3inputs-batch_normalization_31/ReadVariableOp:value:0/batch_normalization_31/ReadVariableOp_1:value:0>batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_31/FusedBatchNormV3?
%batch_normalization_31/AssignNewValueAssignVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource4batch_normalization_31/FusedBatchNormV3:batch_mean:07^batch_normalization_31/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_31/AssignNewValue?
'batch_normalization_31/AssignNewValue_1AssignVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_31/FusedBatchNormV3:batch_variance:09^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_31/AssignNewValue_1?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_60/Conv2D/ReadVariableOp?
conv2d_60/Conv2DConv2D+batch_normalization_31/FusedBatchNormV3:y:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_60/Conv2D?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_60/BiasAdd~
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_60/Relu?
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_61/Conv2D/ReadVariableOp?
conv2d_61/Conv2DConv2Dconv2d_60/Relu:activations:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_61/Conv2D?
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp?
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_61/BiasAdd~
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_61/Relu?
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_32/ReadVariableOp?
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_32/ReadVariableOp_1?
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3conv2d_61/Relu:activations:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_32/FusedBatchNormV3?
%batch_normalization_32/AssignNewValueAssignVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource4batch_normalization_32/FusedBatchNormV3:batch_mean:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_32/AssignNewValue?
'batch_normalization_32/AssignNewValue_1AssignVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_32/FusedBatchNormV3:batch_variance:09^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_32/AssignNewValue_1?
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_62/Conv2D/ReadVariableOp?
conv2d_62/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_62/Conv2D?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_62/BiasAdd~
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_62/Relu?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_63/Conv2D/ReadVariableOp?
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_63/Conv2D?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_63/BiasAdd~
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_63/Relus
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshapeconv2d_63/Relu:activations:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_8/Reshapev
IdentityIdentityflatten_8/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp&^batch_normalization_31/AssignNewValue(^batch_normalization_31/AssignNewValue_17^batch_normalization_31/FusedBatchNormV3/ReadVariableOp9^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_31/ReadVariableOp(^batch_normalization_31/ReadVariableOp_1&^batch_normalization_32/AssignNewValue(^batch_normalization_32/AssignNewValue_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 2N
%batch_normalization_31/AssignNewValue%batch_normalization_31/AssignNewValue2R
'batch_normalization_31/AssignNewValue_1'batch_normalization_31/AssignNewValue_12p
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp6batch_normalization_31/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_18batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_31/ReadVariableOp%batch_normalization_31/ReadVariableOp2R
'batch_normalization_31/ReadVariableOp_1'batch_normalization_31/ReadVariableOp_12N
%batch_normalization_32/AssignNewValue%batch_normalization_32/AssignNewValue2R
'batch_normalization_32/AssignNewValue_1'batch_normalization_32/AssignNewValue_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_5680116	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_56776172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????@@

_user_specified_nameinput
??
?%
"__inference__wrapped_model_5677617	
inputP
Bautoencoder_encoder_batch_normalization_31_readvariableop_resource:R
Dautoencoder_encoder_batch_normalization_31_readvariableop_1_resource:a
Sautoencoder_encoder_batch_normalization_31_fusedbatchnormv3_readvariableop_resource:c
Uautoencoder_encoder_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:V
<autoencoder_encoder_conv2d_60_conv2d_readvariableop_resource:K
=autoencoder_encoder_conv2d_60_biasadd_readvariableop_resource:V
<autoencoder_encoder_conv2d_61_conv2d_readvariableop_resource:K
=autoencoder_encoder_conv2d_61_biasadd_readvariableop_resource:P
Bautoencoder_encoder_batch_normalization_32_readvariableop_resource:R
Dautoencoder_encoder_batch_normalization_32_readvariableop_1_resource:a
Sautoencoder_encoder_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:c
Uautoencoder_encoder_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:V
<autoencoder_encoder_conv2d_62_conv2d_readvariableop_resource:K
=autoencoder_encoder_conv2d_62_biasadd_readvariableop_resource:V
<autoencoder_encoder_conv2d_63_conv2d_readvariableop_resource:K
=autoencoder_encoder_conv2d_63_biasadd_readvariableop_resource:j
Pautoencoder_decoder_conv2d_transpose_57_conv2d_transpose_readvariableop_resource:U
Gautoencoder_decoder_conv2d_transpose_57_biasadd_readvariableop_resource:j
Pautoencoder_decoder_conv2d_transpose_58_conv2d_transpose_readvariableop_resource:U
Gautoencoder_decoder_conv2d_transpose_58_biasadd_readvariableop_resource:P
Bautoencoder_decoder_batch_normalization_33_readvariableop_resource:R
Dautoencoder_decoder_batch_normalization_33_readvariableop_1_resource:a
Sautoencoder_decoder_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:c
Uautoencoder_decoder_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:j
Pautoencoder_decoder_conv2d_transpose_59_conv2d_transpose_readvariableop_resource:U
Gautoencoder_decoder_conv2d_transpose_59_biasadd_readvariableop_resource:j
Pautoencoder_decoder_conv2d_transpose_60_conv2d_transpose_readvariableop_resource:U
Gautoencoder_decoder_conv2d_transpose_60_biasadd_readvariableop_resource:j
Pautoencoder_decoder_conv2d_transpose_61_conv2d_transpose_readvariableop_resource:U
Gautoencoder_decoder_conv2d_transpose_61_biasadd_readvariableop_resource:
identity??Jautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?Lautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?9autoencoder/decoder/batch_normalization_33/ReadVariableOp?;autoencoder/decoder/batch_normalization_33/ReadVariableOp_1?>autoencoder/decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp?Gautoencoder/decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp?>autoencoder/decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp?Gautoencoder/decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp?>autoencoder/decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp?Gautoencoder/decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp?>autoencoder/decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp?Gautoencoder/decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp?>autoencoder/decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp?Gautoencoder/decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp?Jautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?Lautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?9autoencoder/encoder/batch_normalization_31/ReadVariableOp?;autoencoder/encoder/batch_normalization_31/ReadVariableOp_1?Jautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?Lautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?9autoencoder/encoder/batch_normalization_32/ReadVariableOp?;autoencoder/encoder/batch_normalization_32/ReadVariableOp_1?4autoencoder/encoder/conv2d_60/BiasAdd/ReadVariableOp?3autoencoder/encoder/conv2d_60/Conv2D/ReadVariableOp?4autoencoder/encoder/conv2d_61/BiasAdd/ReadVariableOp?3autoencoder/encoder/conv2d_61/Conv2D/ReadVariableOp?4autoencoder/encoder/conv2d_62/BiasAdd/ReadVariableOp?3autoencoder/encoder/conv2d_62/Conv2D/ReadVariableOp?4autoencoder/encoder/conv2d_63/BiasAdd/ReadVariableOp?3autoencoder/encoder/conv2d_63/Conv2D/ReadVariableOp?
9autoencoder/encoder/batch_normalization_31/ReadVariableOpReadVariableOpBautoencoder_encoder_batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype02;
9autoencoder/encoder/batch_normalization_31/ReadVariableOp?
;autoencoder/encoder/batch_normalization_31/ReadVariableOp_1ReadVariableOpDautoencoder_encoder_batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;autoencoder/encoder/batch_normalization_31/ReadVariableOp_1?
Jautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOpSautoencoder_encoder_batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
Lautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUautoencoder_encoder_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02N
Lautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
;autoencoder/encoder/batch_normalization_31/FusedBatchNormV3FusedBatchNormV3inputAautoencoder/encoder/batch_normalization_31/ReadVariableOp:value:0Cautoencoder/encoder/batch_normalization_31/ReadVariableOp_1:value:0Rautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0Tautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2=
;autoencoder/encoder/batch_normalization_31/FusedBatchNormV3?
3autoencoder/encoder/conv2d_60/Conv2D/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3autoencoder/encoder/conv2d_60/Conv2D/ReadVariableOp?
$autoencoder/encoder/conv2d_60/Conv2DConv2D?autoencoder/encoder/batch_normalization_31/FusedBatchNormV3:y:0;autoencoder/encoder/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2&
$autoencoder/encoder/conv2d_60/Conv2D?
4autoencoder/encoder/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_encoder_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4autoencoder/encoder/conv2d_60/BiasAdd/ReadVariableOp?
%autoencoder/encoder/conv2d_60/BiasAddBiasAdd-autoencoder/encoder/conv2d_60/Conv2D:output:0<autoencoder/encoder/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2'
%autoencoder/encoder/conv2d_60/BiasAdd?
"autoencoder/encoder/conv2d_60/ReluRelu.autoencoder/encoder/conv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2$
"autoencoder/encoder/conv2d_60/Relu?
3autoencoder/encoder/conv2d_61/Conv2D/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3autoencoder/encoder/conv2d_61/Conv2D/ReadVariableOp?
$autoencoder/encoder/conv2d_61/Conv2DConv2D0autoencoder/encoder/conv2d_60/Relu:activations:0;autoencoder/encoder/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$autoencoder/encoder/conv2d_61/Conv2D?
4autoencoder/encoder/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_encoder_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4autoencoder/encoder/conv2d_61/BiasAdd/ReadVariableOp?
%autoencoder/encoder/conv2d_61/BiasAddBiasAdd-autoencoder/encoder/conv2d_61/Conv2D:output:0<autoencoder/encoder/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2'
%autoencoder/encoder/conv2d_61/BiasAdd?
"autoencoder/encoder/conv2d_61/ReluRelu.autoencoder/encoder/conv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2$
"autoencoder/encoder/conv2d_61/Relu?
9autoencoder/encoder/batch_normalization_32/ReadVariableOpReadVariableOpBautoencoder_encoder_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype02;
9autoencoder/encoder/batch_normalization_32/ReadVariableOp?
;autoencoder/encoder/batch_normalization_32/ReadVariableOp_1ReadVariableOpDautoencoder_encoder_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;autoencoder/encoder/batch_normalization_32/ReadVariableOp_1?
Jautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpSautoencoder_encoder_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
Lautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUautoencoder_encoder_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02N
Lautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
;autoencoder/encoder/batch_normalization_32/FusedBatchNormV3FusedBatchNormV30autoencoder/encoder/conv2d_61/Relu:activations:0Aautoencoder/encoder/batch_normalization_32/ReadVariableOp:value:0Cautoencoder/encoder/batch_normalization_32/ReadVariableOp_1:value:0Rautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Tautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2=
;autoencoder/encoder/batch_normalization_32/FusedBatchNormV3?
3autoencoder/encoder/conv2d_62/Conv2D/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3autoencoder/encoder/conv2d_62/Conv2D/ReadVariableOp?
$autoencoder/encoder/conv2d_62/Conv2DConv2D?autoencoder/encoder/batch_normalization_32/FusedBatchNormV3:y:0;autoencoder/encoder/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$autoencoder/encoder/conv2d_62/Conv2D?
4autoencoder/encoder/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_encoder_conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4autoencoder/encoder/conv2d_62/BiasAdd/ReadVariableOp?
%autoencoder/encoder/conv2d_62/BiasAddBiasAdd-autoencoder/encoder/conv2d_62/Conv2D:output:0<autoencoder/encoder/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2'
%autoencoder/encoder/conv2d_62/BiasAdd?
"autoencoder/encoder/conv2d_62/ReluRelu.autoencoder/encoder/conv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2$
"autoencoder/encoder/conv2d_62/Relu?
3autoencoder/encoder/conv2d_63/Conv2D/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype025
3autoencoder/encoder/conv2d_63/Conv2D/ReadVariableOp?
$autoencoder/encoder/conv2d_63/Conv2DConv2D0autoencoder/encoder/conv2d_62/Relu:activations:0;autoencoder/encoder/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$autoencoder/encoder/conv2d_63/Conv2D?
4autoencoder/encoder/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp=autoencoder_encoder_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4autoencoder/encoder/conv2d_63/BiasAdd/ReadVariableOp?
%autoencoder/encoder/conv2d_63/BiasAddBiasAdd-autoencoder/encoder/conv2d_63/Conv2D:output:0<autoencoder/encoder/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2'
%autoencoder/encoder/conv2d_63/BiasAdd?
"autoencoder/encoder/conv2d_63/ReluRelu.autoencoder/encoder/conv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"autoencoder/encoder/conv2d_63/Relu?
#autoencoder/encoder/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2%
#autoencoder/encoder/flatten_8/Const?
%autoencoder/encoder/flatten_8/ReshapeReshape0autoencoder/encoder/conv2d_63/Relu:activations:0,autoencoder/encoder/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2'
%autoencoder/encoder/flatten_8/Reshape?
#autoencoder/decoder/reshape_8/ShapeShape.autoencoder/encoder/flatten_8/Reshape:output:0*
T0*
_output_shapes
:2%
#autoencoder/decoder/reshape_8/Shape?
1autoencoder/decoder/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1autoencoder/decoder/reshape_8/strided_slice/stack?
3autoencoder/decoder/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3autoencoder/decoder/reshape_8/strided_slice/stack_1?
3autoencoder/decoder/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3autoencoder/decoder/reshape_8/strided_slice/stack_2?
+autoencoder/decoder/reshape_8/strided_sliceStridedSlice,autoencoder/decoder/reshape_8/Shape:output:0:autoencoder/decoder/reshape_8/strided_slice/stack:output:0<autoencoder/decoder/reshape_8/strided_slice/stack_1:output:0<autoencoder/decoder/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+autoencoder/decoder/reshape_8/strided_slice?
-autoencoder/decoder/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-autoencoder/decoder/reshape_8/Reshape/shape/1?
-autoencoder/decoder/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-autoencoder/decoder/reshape_8/Reshape/shape/2?
-autoencoder/decoder/reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-autoencoder/decoder/reshape_8/Reshape/shape/3?
+autoencoder/decoder/reshape_8/Reshape/shapePack4autoencoder/decoder/reshape_8/strided_slice:output:06autoencoder/decoder/reshape_8/Reshape/shape/1:output:06autoencoder/decoder/reshape_8/Reshape/shape/2:output:06autoencoder/decoder/reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+autoencoder/decoder/reshape_8/Reshape/shape?
%autoencoder/decoder/reshape_8/ReshapeReshape.autoencoder/encoder/flatten_8/Reshape:output:04autoencoder/decoder/reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2'
%autoencoder/decoder/reshape_8/Reshape?
-autoencoder/decoder/conv2d_transpose_57/ShapeShape.autoencoder/decoder/reshape_8/Reshape:output:0*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_57/Shape?
;autoencoder/decoder/conv2d_transpose_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;autoencoder/decoder/conv2d_transpose_57/strided_slice/stack?
=autoencoder/decoder/conv2d_transpose_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_57/strided_slice/stack_1?
=autoencoder/decoder/conv2d_transpose_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_57/strided_slice/stack_2?
5autoencoder/decoder/conv2d_transpose_57/strided_sliceStridedSlice6autoencoder/decoder/conv2d_transpose_57/Shape:output:0Dautoencoder/decoder/conv2d_transpose_57/strided_slice/stack:output:0Fautoencoder/decoder/conv2d_transpose_57/strided_slice/stack_1:output:0Fautoencoder/decoder/conv2d_transpose_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5autoencoder/decoder/conv2d_transpose_57/strided_slice?
/autoencoder/decoder/conv2d_transpose_57/stack/1Const*
_output_shapes
: *
dtype0*
value	B :21
/autoencoder/decoder/conv2d_transpose_57/stack/1?
/autoencoder/decoder/conv2d_transpose_57/stack/2Const*
_output_shapes
: *
dtype0*
value	B :21
/autoencoder/decoder/conv2d_transpose_57/stack/2?
/autoencoder/decoder/conv2d_transpose_57/stack/3Const*
_output_shapes
: *
dtype0*
value	B :21
/autoencoder/decoder/conv2d_transpose_57/stack/3?
-autoencoder/decoder/conv2d_transpose_57/stackPack>autoencoder/decoder/conv2d_transpose_57/strided_slice:output:08autoencoder/decoder/conv2d_transpose_57/stack/1:output:08autoencoder/decoder/conv2d_transpose_57/stack/2:output:08autoencoder/decoder/conv2d_transpose_57/stack/3:output:0*
N*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_57/stack?
=autoencoder/decoder/conv2d_transpose_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=autoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack?
?autoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack_1?
?autoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack_2?
7autoencoder/decoder/conv2d_transpose_57/strided_slice_1StridedSlice6autoencoder/decoder/conv2d_transpose_57/stack:output:0Fautoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack:output:0Hautoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack_1:output:0Hautoencoder/decoder/conv2d_transpose_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7autoencoder/decoder/conv2d_transpose_57/strided_slice_1?
Gautoencoder/decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOpReadVariableOpPautoencoder_decoder_conv2d_transpose_57_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02I
Gautoencoder/decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp?
8autoencoder/decoder/conv2d_transpose_57/conv2d_transposeConv2DBackpropInput6autoencoder/decoder/conv2d_transpose_57/stack:output:0Oautoencoder/decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp:value:0.autoencoder/decoder/reshape_8/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2:
8autoencoder/decoder/conv2d_transpose_57/conv2d_transpose?
>autoencoder/decoder/conv2d_transpose_57/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_decoder_conv2d_transpose_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>autoencoder/decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp?
/autoencoder/decoder/conv2d_transpose_57/BiasAddBiasAddAautoencoder/decoder/conv2d_transpose_57/conv2d_transpose:output:0Fautoencoder/decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????21
/autoencoder/decoder/conv2d_transpose_57/BiasAdd?
,autoencoder/decoder/conv2d_transpose_57/ReluRelu8autoencoder/decoder/conv2d_transpose_57/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2.
,autoencoder/decoder/conv2d_transpose_57/Relu?
-autoencoder/decoder/conv2d_transpose_58/ShapeShape:autoencoder/decoder/conv2d_transpose_57/Relu:activations:0*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_58/Shape?
;autoencoder/decoder/conv2d_transpose_58/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;autoencoder/decoder/conv2d_transpose_58/strided_slice/stack?
=autoencoder/decoder/conv2d_transpose_58/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_58/strided_slice/stack_1?
=autoencoder/decoder/conv2d_transpose_58/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_58/strided_slice/stack_2?
5autoencoder/decoder/conv2d_transpose_58/strided_sliceStridedSlice6autoencoder/decoder/conv2d_transpose_58/Shape:output:0Dautoencoder/decoder/conv2d_transpose_58/strided_slice/stack:output:0Fautoencoder/decoder/conv2d_transpose_58/strided_slice/stack_1:output:0Fautoencoder/decoder/conv2d_transpose_58/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5autoencoder/decoder/conv2d_transpose_58/strided_slice?
/autoencoder/decoder/conv2d_transpose_58/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/decoder/conv2d_transpose_58/stack/1?
/autoencoder/decoder/conv2d_transpose_58/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/decoder/conv2d_transpose_58/stack/2?
/autoencoder/decoder/conv2d_transpose_58/stack/3Const*
_output_shapes
: *
dtype0*
value	B :21
/autoencoder/decoder/conv2d_transpose_58/stack/3?
-autoencoder/decoder/conv2d_transpose_58/stackPack>autoencoder/decoder/conv2d_transpose_58/strided_slice:output:08autoencoder/decoder/conv2d_transpose_58/stack/1:output:08autoencoder/decoder/conv2d_transpose_58/stack/2:output:08autoencoder/decoder/conv2d_transpose_58/stack/3:output:0*
N*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_58/stack?
=autoencoder/decoder/conv2d_transpose_58/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=autoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack?
?autoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack_1?
?autoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack_2?
7autoencoder/decoder/conv2d_transpose_58/strided_slice_1StridedSlice6autoencoder/decoder/conv2d_transpose_58/stack:output:0Fautoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack:output:0Hautoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack_1:output:0Hautoencoder/decoder/conv2d_transpose_58/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7autoencoder/decoder/conv2d_transpose_58/strided_slice_1?
Gautoencoder/decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOpReadVariableOpPautoencoder_decoder_conv2d_transpose_58_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02I
Gautoencoder/decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp?
8autoencoder/decoder/conv2d_transpose_58/conv2d_transposeConv2DBackpropInput6autoencoder/decoder/conv2d_transpose_58/stack:output:0Oautoencoder/decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp:value:0:autoencoder/decoder/conv2d_transpose_57/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2:
8autoencoder/decoder/conv2d_transpose_58/conv2d_transpose?
>autoencoder/decoder/conv2d_transpose_58/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_decoder_conv2d_transpose_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>autoencoder/decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp?
/autoencoder/decoder/conv2d_transpose_58/BiasAddBiasAddAautoencoder/decoder/conv2d_transpose_58/conv2d_transpose:output:0Fautoencoder/decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  21
/autoencoder/decoder/conv2d_transpose_58/BiasAdd?
,autoencoder/decoder/conv2d_transpose_58/ReluRelu8autoencoder/decoder/conv2d_transpose_58/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2.
,autoencoder/decoder/conv2d_transpose_58/Relu?
9autoencoder/decoder/batch_normalization_33/ReadVariableOpReadVariableOpBautoencoder_decoder_batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype02;
9autoencoder/decoder/batch_normalization_33/ReadVariableOp?
;autoencoder/decoder/batch_normalization_33/ReadVariableOp_1ReadVariableOpDautoencoder_decoder_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;autoencoder/decoder/batch_normalization_33/ReadVariableOp_1?
Jautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpSautoencoder_decoder_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02L
Jautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
Lautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpUautoencoder_decoder_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02N
Lautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
;autoencoder/decoder/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3:autoencoder/decoder/conv2d_transpose_58/Relu:activations:0Aautoencoder/decoder/batch_normalization_33/ReadVariableOp:value:0Cautoencoder/decoder/batch_normalization_33/ReadVariableOp_1:value:0Rautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Tautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2=
;autoencoder/decoder/batch_normalization_33/FusedBatchNormV3?
-autoencoder/decoder/conv2d_transpose_59/ShapeShape?autoencoder/decoder/batch_normalization_33/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_59/Shape?
;autoencoder/decoder/conv2d_transpose_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;autoencoder/decoder/conv2d_transpose_59/strided_slice/stack?
=autoencoder/decoder/conv2d_transpose_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_59/strided_slice/stack_1?
=autoencoder/decoder/conv2d_transpose_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_59/strided_slice/stack_2?
5autoencoder/decoder/conv2d_transpose_59/strided_sliceStridedSlice6autoencoder/decoder/conv2d_transpose_59/Shape:output:0Dautoencoder/decoder/conv2d_transpose_59/strided_slice/stack:output:0Fautoencoder/decoder/conv2d_transpose_59/strided_slice/stack_1:output:0Fautoencoder/decoder/conv2d_transpose_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5autoencoder/decoder/conv2d_transpose_59/strided_slice?
/autoencoder/decoder/conv2d_transpose_59/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/decoder/conv2d_transpose_59/stack/1?
/autoencoder/decoder/conv2d_transpose_59/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 21
/autoencoder/decoder/conv2d_transpose_59/stack/2?
/autoencoder/decoder/conv2d_transpose_59/stack/3Const*
_output_shapes
: *
dtype0*
value	B :21
/autoencoder/decoder/conv2d_transpose_59/stack/3?
-autoencoder/decoder/conv2d_transpose_59/stackPack>autoencoder/decoder/conv2d_transpose_59/strided_slice:output:08autoencoder/decoder/conv2d_transpose_59/stack/1:output:08autoencoder/decoder/conv2d_transpose_59/stack/2:output:08autoencoder/decoder/conv2d_transpose_59/stack/3:output:0*
N*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_59/stack?
=autoencoder/decoder/conv2d_transpose_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=autoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack?
?autoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack_1?
?autoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack_2?
7autoencoder/decoder/conv2d_transpose_59/strided_slice_1StridedSlice6autoencoder/decoder/conv2d_transpose_59/stack:output:0Fautoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack:output:0Hautoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack_1:output:0Hautoencoder/decoder/conv2d_transpose_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7autoencoder/decoder/conv2d_transpose_59/strided_slice_1?
Gautoencoder/decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOpReadVariableOpPautoencoder_decoder_conv2d_transpose_59_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02I
Gautoencoder/decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp?
8autoencoder/decoder/conv2d_transpose_59/conv2d_transposeConv2DBackpropInput6autoencoder/decoder/conv2d_transpose_59/stack:output:0Oautoencoder/decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp:value:0?autoencoder/decoder/batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2:
8autoencoder/decoder/conv2d_transpose_59/conv2d_transpose?
>autoencoder/decoder/conv2d_transpose_59/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_decoder_conv2d_transpose_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>autoencoder/decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp?
/autoencoder/decoder/conv2d_transpose_59/BiasAddBiasAddAautoencoder/decoder/conv2d_transpose_59/conv2d_transpose:output:0Fautoencoder/decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  21
/autoencoder/decoder/conv2d_transpose_59/BiasAdd?
,autoencoder/decoder/conv2d_transpose_59/ReluRelu8autoencoder/decoder/conv2d_transpose_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2.
,autoencoder/decoder/conv2d_transpose_59/Relu?
-autoencoder/decoder/conv2d_transpose_60/ShapeShape:autoencoder/decoder/conv2d_transpose_59/Relu:activations:0*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_60/Shape?
;autoencoder/decoder/conv2d_transpose_60/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;autoencoder/decoder/conv2d_transpose_60/strided_slice/stack?
=autoencoder/decoder/conv2d_transpose_60/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_60/strided_slice/stack_1?
=autoencoder/decoder/conv2d_transpose_60/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_60/strided_slice/stack_2?
5autoencoder/decoder/conv2d_transpose_60/strided_sliceStridedSlice6autoencoder/decoder/conv2d_transpose_60/Shape:output:0Dautoencoder/decoder/conv2d_transpose_60/strided_slice/stack:output:0Fautoencoder/decoder/conv2d_transpose_60/strided_slice/stack_1:output:0Fautoencoder/decoder/conv2d_transpose_60/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5autoencoder/decoder/conv2d_transpose_60/strided_slice?
/autoencoder/decoder/conv2d_transpose_60/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@21
/autoencoder/decoder/conv2d_transpose_60/stack/1?
/autoencoder/decoder/conv2d_transpose_60/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@21
/autoencoder/decoder/conv2d_transpose_60/stack/2?
/autoencoder/decoder/conv2d_transpose_60/stack/3Const*
_output_shapes
: *
dtype0*
value	B :21
/autoencoder/decoder/conv2d_transpose_60/stack/3?
-autoencoder/decoder/conv2d_transpose_60/stackPack>autoencoder/decoder/conv2d_transpose_60/strided_slice:output:08autoencoder/decoder/conv2d_transpose_60/stack/1:output:08autoencoder/decoder/conv2d_transpose_60/stack/2:output:08autoencoder/decoder/conv2d_transpose_60/stack/3:output:0*
N*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_60/stack?
=autoencoder/decoder/conv2d_transpose_60/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=autoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack?
?autoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack_1?
?autoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack_2?
7autoencoder/decoder/conv2d_transpose_60/strided_slice_1StridedSlice6autoencoder/decoder/conv2d_transpose_60/stack:output:0Fautoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack:output:0Hautoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack_1:output:0Hautoencoder/decoder/conv2d_transpose_60/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7autoencoder/decoder/conv2d_transpose_60/strided_slice_1?
Gautoencoder/decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOpReadVariableOpPautoencoder_decoder_conv2d_transpose_60_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02I
Gautoencoder/decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp?
8autoencoder/decoder/conv2d_transpose_60/conv2d_transposeConv2DBackpropInput6autoencoder/decoder/conv2d_transpose_60/stack:output:0Oautoencoder/decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp:value:0:autoencoder/decoder/conv2d_transpose_59/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2:
8autoencoder/decoder/conv2d_transpose_60/conv2d_transpose?
>autoencoder/decoder/conv2d_transpose_60/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_decoder_conv2d_transpose_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>autoencoder/decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp?
/autoencoder/decoder/conv2d_transpose_60/BiasAddBiasAddAautoencoder/decoder/conv2d_transpose_60/conv2d_transpose:output:0Fautoencoder/decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@21
/autoencoder/decoder/conv2d_transpose_60/BiasAdd?
,autoencoder/decoder/conv2d_transpose_60/ReluRelu8autoencoder/decoder/conv2d_transpose_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2.
,autoencoder/decoder/conv2d_transpose_60/Relu?
-autoencoder/decoder/conv2d_transpose_61/ShapeShape:autoencoder/decoder/conv2d_transpose_60/Relu:activations:0*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_61/Shape?
;autoencoder/decoder/conv2d_transpose_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;autoencoder/decoder/conv2d_transpose_61/strided_slice/stack?
=autoencoder/decoder/conv2d_transpose_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_61/strided_slice/stack_1?
=autoencoder/decoder/conv2d_transpose_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=autoencoder/decoder/conv2d_transpose_61/strided_slice/stack_2?
5autoencoder/decoder/conv2d_transpose_61/strided_sliceStridedSlice6autoencoder/decoder/conv2d_transpose_61/Shape:output:0Dautoencoder/decoder/conv2d_transpose_61/strided_slice/stack:output:0Fautoencoder/decoder/conv2d_transpose_61/strided_slice/stack_1:output:0Fautoencoder/decoder/conv2d_transpose_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5autoencoder/decoder/conv2d_transpose_61/strided_slice?
/autoencoder/decoder/conv2d_transpose_61/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@21
/autoencoder/decoder/conv2d_transpose_61/stack/1?
/autoencoder/decoder/conv2d_transpose_61/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@21
/autoencoder/decoder/conv2d_transpose_61/stack/2?
/autoencoder/decoder/conv2d_transpose_61/stack/3Const*
_output_shapes
: *
dtype0*
value	B :21
/autoencoder/decoder/conv2d_transpose_61/stack/3?
-autoencoder/decoder/conv2d_transpose_61/stackPack>autoencoder/decoder/conv2d_transpose_61/strided_slice:output:08autoencoder/decoder/conv2d_transpose_61/stack/1:output:08autoencoder/decoder/conv2d_transpose_61/stack/2:output:08autoencoder/decoder/conv2d_transpose_61/stack/3:output:0*
N*
T0*
_output_shapes
:2/
-autoencoder/decoder/conv2d_transpose_61/stack?
=autoencoder/decoder/conv2d_transpose_61/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=autoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack?
?autoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack_1?
?autoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?autoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack_2?
7autoencoder/decoder/conv2d_transpose_61/strided_slice_1StridedSlice6autoencoder/decoder/conv2d_transpose_61/stack:output:0Fautoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack:output:0Hautoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack_1:output:0Hautoencoder/decoder/conv2d_transpose_61/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7autoencoder/decoder/conv2d_transpose_61/strided_slice_1?
Gautoencoder/decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOpReadVariableOpPautoencoder_decoder_conv2d_transpose_61_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02I
Gautoencoder/decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp?
8autoencoder/decoder/conv2d_transpose_61/conv2d_transposeConv2DBackpropInput6autoencoder/decoder/conv2d_transpose_61/stack:output:0Oautoencoder/decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp:value:0:autoencoder/decoder/conv2d_transpose_60/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2:
8autoencoder/decoder/conv2d_transpose_61/conv2d_transpose?
>autoencoder/decoder/conv2d_transpose_61/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_decoder_conv2d_transpose_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02@
>autoencoder/decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp?
/autoencoder/decoder/conv2d_transpose_61/BiasAddBiasAddAautoencoder/decoder/conv2d_transpose_61/conv2d_transpose:output:0Fautoencoder/decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@21
/autoencoder/decoder/conv2d_transpose_61/BiasAdd?
IdentityIdentity8autoencoder/decoder/conv2d_transpose_61/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOpK^autoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOpM^autoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:^autoencoder/decoder/batch_normalization_33/ReadVariableOp<^autoencoder/decoder/batch_normalization_33/ReadVariableOp_1?^autoencoder/decoder/conv2d_transpose_57/BiasAdd/ReadVariableOpH^autoencoder/decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp?^autoencoder/decoder/conv2d_transpose_58/BiasAdd/ReadVariableOpH^autoencoder/decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp?^autoencoder/decoder/conv2d_transpose_59/BiasAdd/ReadVariableOpH^autoencoder/decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp?^autoencoder/decoder/conv2d_transpose_60/BiasAdd/ReadVariableOpH^autoencoder/decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp?^autoencoder/decoder/conv2d_transpose_61/BiasAdd/ReadVariableOpH^autoencoder/decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOpK^autoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOpM^autoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:^autoencoder/encoder/batch_normalization_31/ReadVariableOp<^autoencoder/encoder/batch_normalization_31/ReadVariableOp_1K^autoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOpM^autoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:^autoencoder/encoder/batch_normalization_32/ReadVariableOp<^autoencoder/encoder/batch_normalization_32/ReadVariableOp_15^autoencoder/encoder/conv2d_60/BiasAdd/ReadVariableOp4^autoencoder/encoder/conv2d_60/Conv2D/ReadVariableOp5^autoencoder/encoder/conv2d_61/BiasAdd/ReadVariableOp4^autoencoder/encoder/conv2d_61/Conv2D/ReadVariableOp5^autoencoder/encoder/conv2d_62/BiasAdd/ReadVariableOp4^autoencoder/encoder/conv2d_62/Conv2D/ReadVariableOp5^autoencoder/encoder/conv2d_63/BiasAdd/ReadVariableOp4^autoencoder/encoder/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Jautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOpJautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
Lautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Lautoencoder/decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12v
9autoencoder/decoder/batch_normalization_33/ReadVariableOp9autoencoder/decoder/batch_normalization_33/ReadVariableOp2z
;autoencoder/decoder/batch_normalization_33/ReadVariableOp_1;autoencoder/decoder/batch_normalization_33/ReadVariableOp_12?
>autoencoder/decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp>autoencoder/decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp2?
Gautoencoder/decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOpGautoencoder/decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp2?
>autoencoder/decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp>autoencoder/decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp2?
Gautoencoder/decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOpGautoencoder/decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp2?
>autoencoder/decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp>autoencoder/decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp2?
Gautoencoder/decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOpGautoencoder/decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp2?
>autoencoder/decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp>autoencoder/decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp2?
Gautoencoder/decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOpGautoencoder/decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp2?
>autoencoder/decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp>autoencoder/decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp2?
Gautoencoder/decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOpGautoencoder/decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp2?
Jautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOpJautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp2?
Lautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1Lautoencoder/encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12v
9autoencoder/encoder/batch_normalization_31/ReadVariableOp9autoencoder/encoder/batch_normalization_31/ReadVariableOp2z
;autoencoder/encoder/batch_normalization_31/ReadVariableOp_1;autoencoder/encoder/batch_normalization_31/ReadVariableOp_12?
Jautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOpJautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2?
Lautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Lautoencoder/encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12v
9autoencoder/encoder/batch_normalization_32/ReadVariableOp9autoencoder/encoder/batch_normalization_32/ReadVariableOp2z
;autoencoder/encoder/batch_normalization_32/ReadVariableOp_1;autoencoder/encoder/batch_normalization_32/ReadVariableOp_12l
4autoencoder/encoder/conv2d_60/BiasAdd/ReadVariableOp4autoencoder/encoder/conv2d_60/BiasAdd/ReadVariableOp2j
3autoencoder/encoder/conv2d_60/Conv2D/ReadVariableOp3autoencoder/encoder/conv2d_60/Conv2D/ReadVariableOp2l
4autoencoder/encoder/conv2d_61/BiasAdd/ReadVariableOp4autoencoder/encoder/conv2d_61/BiasAdd/ReadVariableOp2j
3autoencoder/encoder/conv2d_61/Conv2D/ReadVariableOp3autoencoder/encoder/conv2d_61/Conv2D/ReadVariableOp2l
4autoencoder/encoder/conv2d_62/BiasAdd/ReadVariableOp4autoencoder/encoder/conv2d_62/BiasAdd/ReadVariableOp2j
3autoencoder/encoder/conv2d_62/Conv2D/ReadVariableOp3autoencoder/encoder/conv2d_62/Conv2D/ReadVariableOp2l
4autoencoder/encoder/conv2d_63/BiasAdd/ReadVariableOp4autoencoder/encoder/conv2d_63/BiasAdd/ReadVariableOp2j
3autoencoder/encoder/conv2d_63/Conv2D/ReadVariableOp3autoencoder/encoder/conv2d_63/Conv2D/ReadVariableOp:V R
/
_output_shapes
:?????????@@

_user_specified_nameinput
?	
?
8__inference_batch_normalization_32_layer_call_fn_5681324

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_56778092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_5677931

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5678542

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?T
?
D__inference_encoder_layer_call_and_return_conditional_losses_5680752

inputs<
.batch_normalization_31_readvariableop_resource:>
0batch_normalization_31_readvariableop_1_resource:M
?batch_normalization_31_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_60_conv2d_readvariableop_resource:7
)conv2d_60_biasadd_readvariableop_resource:B
(conv2d_61_conv2d_readvariableop_resource:7
)conv2d_61_biasadd_readvariableop_resource:<
.batch_normalization_32_readvariableop_resource:>
0batch_normalization_32_readvariableop_1_resource:M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_62_conv2d_readvariableop_resource:7
)conv2d_62_biasadd_readvariableop_resource:B
(conv2d_63_conv2d_readvariableop_resource:7
)conv2d_63_biasadd_readvariableop_resource:
identity??6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_31/ReadVariableOp?'batch_normalization_31/ReadVariableOp_1?6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_32/ReadVariableOp?'batch_normalization_32/ReadVariableOp_1? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp? conv2d_61/BiasAdd/ReadVariableOp?conv2d_61/Conv2D/ReadVariableOp? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp?
%batch_normalization_31/ReadVariableOpReadVariableOp.batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_31/ReadVariableOp?
'batch_normalization_31/ReadVariableOp_1ReadVariableOp0batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_31/ReadVariableOp_1?
6batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_31/FusedBatchNormV3FusedBatchNormV3inputs-batch_normalization_31/ReadVariableOp:value:0/batch_normalization_31/ReadVariableOp_1:value:0>batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_31/FusedBatchNormV3?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_60/Conv2D/ReadVariableOp?
conv2d_60/Conv2DConv2D+batch_normalization_31/FusedBatchNormV3:y:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_60/Conv2D?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_60/BiasAdd~
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_60/Relu?
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_61/Conv2D/ReadVariableOp?
conv2d_61/Conv2DConv2Dconv2d_60/Relu:activations:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_61/Conv2D?
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp?
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_61/BiasAdd~
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_61/Relu?
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_32/ReadVariableOp?
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_32/ReadVariableOp_1?
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3conv2d_61/Relu:activations:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_32/FusedBatchNormV3?
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_62/Conv2D/ReadVariableOp?
conv2d_62/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_62/Conv2D?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_62/BiasAdd~
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_62/Relu?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_63/Conv2D/ReadVariableOp?
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_63/Conv2D?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_63/BiasAdd~
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_63/Relus
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_8/Const?
flatten_8/ReshapeReshapeconv2d_63/Relu:activations:0flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_8/Reshapev
IdentityIdentityflatten_8/Reshape:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp7^batch_normalization_31/FusedBatchNormV3/ReadVariableOp9^batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_31/ReadVariableOp(^batch_normalization_31/ReadVariableOp_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 2p
6batch_normalization_31/FusedBatchNormV3/ReadVariableOp6batch_normalization_31/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_31/FusedBatchNormV3/ReadVariableOp_18batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_31/ReadVariableOp%batch_normalization_31/ReadVariableOp2R
'batch_normalization_31/ReadVariableOp_1'batch_normalization_31/ReadVariableOp_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?	
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680043	
input
encoder_5679980:
encoder_5679982:
encoder_5679984:
encoder_5679986:)
encoder_5679988:
encoder_5679990:)
encoder_5679992:
encoder_5679994:
encoder_5679996:
encoder_5679998:
encoder_5680000:
encoder_5680002:)
encoder_5680004:
encoder_5680006:)
encoder_5680008:
encoder_5680010:)
decoder_5680013:
decoder_5680015:)
decoder_5680017:
decoder_5680019:
decoder_5680021:
decoder_5680023:
decoder_5680025:
decoder_5680027:)
decoder_5680029:
decoder_5680031:)
decoder_5680033:
decoder_5680035:)
decoder_5680037:
decoder_5680039:
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputencoder_5679980encoder_5679982encoder_5679984encoder_5679986encoder_5679988encoder_5679990encoder_5679992encoder_5679994encoder_5679996encoder_5679998encoder_5680000encoder_5680002encoder_5680004encoder_5680006encoder_5680008encoder_5680010*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56782582!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_5680013decoder_5680015decoder_5680017decoder_5680019decoder_5680021decoder_5680023decoder_5680025decoder_5680027decoder_5680029decoder_5680031decoder_5680033decoder_5680035decoder_5680037decoder_5680039*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56793752!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:V R
/
_output_shapes
:?????????@@

_user_specified_nameinput
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5677639

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_33_layer_call_fn_5681670

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_56786582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?	
H__inference_autoencoder_layer_call_and_return_conditional_losses_5679783

inputs
encoder_5679720:
encoder_5679722:
encoder_5679724:
encoder_5679726:)
encoder_5679728:
encoder_5679730:)
encoder_5679732:
encoder_5679734:
encoder_5679736:
encoder_5679738:
encoder_5679740:
encoder_5679742:)
encoder_5679744:
encoder_5679746:)
encoder_5679748:
encoder_5679750:)
decoder_5679753:
decoder_5679755:)
decoder_5679757:
decoder_5679759:
decoder_5679761:
decoder_5679763:
decoder_5679765:
decoder_5679767:)
decoder_5679769:
decoder_5679771:)
decoder_5679773:
decoder_5679775:)
decoder_5679777:
decoder_5679779:
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_5679720encoder_5679722encoder_5679724encoder_5679726encoder_5679728encoder_5679730encoder_5679732encoder_5679734encoder_5679736encoder_5679738encoder_5679740encoder_5679742encoder_5679744encoder_5679746encoder_5679748encoder_5679750*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56782582!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_5679753decoder_5679755decoder_5679757decoder_5679759decoder_5679761decoder_5679763decoder_5679765decoder_5679767decoder_5679769decoder_5679771decoder_5679773decoder_5679775decoder_5679777decoder_5679779*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56793752!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_32_layer_call_fn_5681337

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_56779542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681386

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_61_layer_call_fn_5681287

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_56779312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?.
?
D__inference_decoder_layer_call_and_return_conditional_losses_5679478
placeholder5
conv2d_transpose_57_5679443:)
conv2d_transpose_57_5679445:5
conv2d_transpose_58_5679448:)
conv2d_transpose_58_5679450:,
batch_normalization_33_5679453:,
batch_normalization_33_5679455:,
batch_normalization_33_5679457:,
batch_normalization_33_5679459:5
conv2d_transpose_59_5679462:)
conv2d_transpose_59_5679464:5
conv2d_transpose_60_5679467:)
conv2d_transpose_60_5679469:5
conv2d_transpose_61_5679472:)
conv2d_transpose_61_5679474:
identity??.batch_normalization_33/StatefulPartitionedCall?+conv2d_transpose_57/StatefulPartitionedCall?+conv2d_transpose_58/StatefulPartitionedCall?+conv2d_transpose_59/StatefulPartitionedCall?+conv2d_transpose_60/StatefulPartitionedCall?+conv2d_transpose_61/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallplaceholder*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_56790012
reshape_8/PartitionedCall?
+conv2d_transpose_57/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_transpose_57_5679443conv2d_transpose_57_5679445*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_56790262-
+conv2d_transpose_57/StatefulPartitionedCall?
+conv2d_transpose_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_57/StatefulPartitionedCall:output:0conv2d_transpose_58_5679448conv2d_transpose_58_5679450*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_56790552-
+conv2d_transpose_58/StatefulPartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_58/StatefulPartitionedCall:output:0batch_normalization_33_5679453batch_normalization_33_5679455batch_normalization_33_5679457batch_normalization_33_5679459*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_567907820
.batch_normalization_33/StatefulPartitionedCall?
+conv2d_transpose_59/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_transpose_59_5679462conv2d_transpose_59_5679464*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_56791112-
+conv2d_transpose_59/StatefulPartitionedCall?
+conv2d_transpose_60/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_59/StatefulPartitionedCall:output:0conv2d_transpose_60_5679467conv2d_transpose_60_5679469*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_56791402-
+conv2d_transpose_60/StatefulPartitionedCall?
+conv2d_transpose_61/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_60/StatefulPartitionedCall:output:0conv2d_transpose_61_5679472conv2d_transpose_61_5679474*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_56791682-
+conv2d_transpose_61/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_61/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall,^conv2d_transpose_57/StatefulPartitionedCall,^conv2d_transpose_58/StatefulPartitionedCall,^conv2d_transpose_59/StatefulPartitionedCall,^conv2d_transpose_60/StatefulPartitionedCall,^conv2d_transpose_61/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2Z
+conv2d_transpose_57/StatefulPartitionedCall+conv2d_transpose_57/StatefulPartitionedCall2Z
+conv2d_transpose_58/StatefulPartitionedCall+conv2d_transpose_58/StatefulPartitionedCall2Z
+conv2d_transpose_59/StatefulPartitionedCall+conv2d_transpose_59/StatefulPartitionedCall2Z
+conv2d_transpose_60/StatefulPartitionedCall+conv2d_transpose_60/StatefulPartitionedCall2Z
+conv2d_transpose_61/StatefulPartitionedCall+conv2d_transpose_61/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoded audio
?
?
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5681568

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681204

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?,
?
D__inference_encoder_layer_call_and_return_conditional_losses_5678373
placeholder,
batch_normalization_31_5678333:,
batch_normalization_31_5678335:,
batch_normalization_31_5678337:,
batch_normalization_31_5678339:+
conv2d_60_5678342:
conv2d_60_5678344:+
conv2d_61_5678347:
conv2d_61_5678349:,
batch_normalization_32_5678352:,
batch_normalization_32_5678354:,
batch_normalization_32_5678356:,
batch_normalization_32_5678358:+
conv2d_62_5678361:
conv2d_62_5678363:+
conv2d_63_5678366:
conv2d_63_5678368:
identity??.batch_normalization_31/StatefulPartitionedCall?.batch_normalization_32/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCallplaceholderbatch_normalization_31_5678333batch_normalization_31_5678335batch_normalization_31_5678337batch_normalization_31_5678339*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_567789320
.batch_normalization_31/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_60_5678342conv2d_60_5678344*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_56779142#
!conv2d_60/StatefulPartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0conv2d_61_5678347conv2d_61_5678349*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_56779312#
!conv2d_61/StatefulPartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_32_5678352batch_normalization_32_5678354batch_normalization_32_5678356batch_normalization_32_5678358*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_567795420
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_62_5678361conv2d_62_5678363*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_56779752#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_5678366conv2d_63_5678368*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_56779922#
!conv2d_63/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_56780042
flatten_8/PartitionedCall~
IdentityIdentity"flatten_8/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_31/StatefulPartitionedCall/^batch_normalization_32/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@
(
_user_specified_nameoriginal audio
?
?
5__inference_conv2d_transpose_61_layer_call_fn_5681929

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_56789312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
D__inference_decoder_layer_call_and_return_conditional_losses_5681007

inputsV
<conv2d_transpose_57_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_57_biasadd_readvariableop_resource:V
<conv2d_transpose_58_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_58_biasadd_readvariableop_resource:<
.batch_normalization_33_readvariableop_resource:>
0batch_normalization_33_readvariableop_1_resource:M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:V
<conv2d_transpose_59_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_59_biasadd_readvariableop_resource:V
<conv2d_transpose_60_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_60_biasadd_readvariableop_resource:V
<conv2d_transpose_61_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_61_biasadd_readvariableop_resource:
identity??6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?*conv2d_transpose_57/BiasAdd/ReadVariableOp?3conv2d_transpose_57/conv2d_transpose/ReadVariableOp?*conv2d_transpose_58/BiasAdd/ReadVariableOp?3conv2d_transpose_58/conv2d_transpose/ReadVariableOp?*conv2d_transpose_59/BiasAdd/ReadVariableOp?3conv2d_transpose_59/conv2d_transpose/ReadVariableOp?*conv2d_transpose_60/BiasAdd/ReadVariableOp?3conv2d_transpose_60/conv2d_transpose/ReadVariableOp?*conv2d_transpose_61/BiasAdd/ReadVariableOp?3conv2d_transpose_61/conv2d_transpose/ReadVariableOpX
reshape_8/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2x
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeinputs reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_8/Reshape?
conv2d_transpose_57/ShapeShapereshape_8/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_57/Shape?
'conv2d_transpose_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_57/strided_slice/stack?
)conv2d_transpose_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_57/strided_slice/stack_1?
)conv2d_transpose_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_57/strided_slice/stack_2?
!conv2d_transpose_57/strided_sliceStridedSlice"conv2d_transpose_57/Shape:output:00conv2d_transpose_57/strided_slice/stack:output:02conv2d_transpose_57/strided_slice/stack_1:output:02conv2d_transpose_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_57/strided_slice|
conv2d_transpose_57/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_57/stack/1|
conv2d_transpose_57/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_57/stack/2|
conv2d_transpose_57/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_57/stack/3?
conv2d_transpose_57/stackPack*conv2d_transpose_57/strided_slice:output:0$conv2d_transpose_57/stack/1:output:0$conv2d_transpose_57/stack/2:output:0$conv2d_transpose_57/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_57/stack?
)conv2d_transpose_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_57/strided_slice_1/stack?
+conv2d_transpose_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_57/strided_slice_1/stack_1?
+conv2d_transpose_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_57/strided_slice_1/stack_2?
#conv2d_transpose_57/strided_slice_1StridedSlice"conv2d_transpose_57/stack:output:02conv2d_transpose_57/strided_slice_1/stack:output:04conv2d_transpose_57/strided_slice_1/stack_1:output:04conv2d_transpose_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_57/strided_slice_1?
3conv2d_transpose_57/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_57_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_57/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_57/conv2d_transposeConv2DBackpropInput"conv2d_transpose_57/stack:output:0;conv2d_transpose_57/conv2d_transpose/ReadVariableOp:value:0reshape_8/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_57/conv2d_transpose?
*conv2d_transpose_57/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_57/BiasAdd/ReadVariableOp?
conv2d_transpose_57/BiasAddBiasAdd-conv2d_transpose_57/conv2d_transpose:output:02conv2d_transpose_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_57/BiasAdd?
conv2d_transpose_57/ReluRelu$conv2d_transpose_57/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_57/Relu?
conv2d_transpose_58/ShapeShape&conv2d_transpose_57/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_58/Shape?
'conv2d_transpose_58/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_58/strided_slice/stack?
)conv2d_transpose_58/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_58/strided_slice/stack_1?
)conv2d_transpose_58/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_58/strided_slice/stack_2?
!conv2d_transpose_58/strided_sliceStridedSlice"conv2d_transpose_58/Shape:output:00conv2d_transpose_58/strided_slice/stack:output:02conv2d_transpose_58/strided_slice/stack_1:output:02conv2d_transpose_58/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_58/strided_slice|
conv2d_transpose_58/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_58/stack/1|
conv2d_transpose_58/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_58/stack/2|
conv2d_transpose_58/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_58/stack/3?
conv2d_transpose_58/stackPack*conv2d_transpose_58/strided_slice:output:0$conv2d_transpose_58/stack/1:output:0$conv2d_transpose_58/stack/2:output:0$conv2d_transpose_58/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_58/stack?
)conv2d_transpose_58/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_58/strided_slice_1/stack?
+conv2d_transpose_58/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_58/strided_slice_1/stack_1?
+conv2d_transpose_58/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_58/strided_slice_1/stack_2?
#conv2d_transpose_58/strided_slice_1StridedSlice"conv2d_transpose_58/stack:output:02conv2d_transpose_58/strided_slice_1/stack:output:04conv2d_transpose_58/strided_slice_1/stack_1:output:04conv2d_transpose_58/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_58/strided_slice_1?
3conv2d_transpose_58/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_58_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_58/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_58/conv2d_transposeConv2DBackpropInput"conv2d_transpose_58/stack:output:0;conv2d_transpose_58/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_57/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$conv2d_transpose_58/conv2d_transpose?
*conv2d_transpose_58/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_58/BiasAdd/ReadVariableOp?
conv2d_transpose_58/BiasAddBiasAdd-conv2d_transpose_58/conv2d_transpose:output:02conv2d_transpose_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_58/BiasAdd?
conv2d_transpose_58/ReluRelu$conv2d_transpose_58/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_58/Relu?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_33/ReadVariableOp?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_33/ReadVariableOp_1?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_58/Relu:activations:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2)
'batch_normalization_33/FusedBatchNormV3?
conv2d_transpose_59/ShapeShape+batch_normalization_33/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_59/Shape?
'conv2d_transpose_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_59/strided_slice/stack?
)conv2d_transpose_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_59/strided_slice/stack_1?
)conv2d_transpose_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_59/strided_slice/stack_2?
!conv2d_transpose_59/strided_sliceStridedSlice"conv2d_transpose_59/Shape:output:00conv2d_transpose_59/strided_slice/stack:output:02conv2d_transpose_59/strided_slice/stack_1:output:02conv2d_transpose_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_59/strided_slice|
conv2d_transpose_59/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_59/stack/1|
conv2d_transpose_59/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_59/stack/2|
conv2d_transpose_59/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_59/stack/3?
conv2d_transpose_59/stackPack*conv2d_transpose_59/strided_slice:output:0$conv2d_transpose_59/stack/1:output:0$conv2d_transpose_59/stack/2:output:0$conv2d_transpose_59/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_59/stack?
)conv2d_transpose_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_59/strided_slice_1/stack?
+conv2d_transpose_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_59/strided_slice_1/stack_1?
+conv2d_transpose_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_59/strided_slice_1/stack_2?
#conv2d_transpose_59/strided_slice_1StridedSlice"conv2d_transpose_59/stack:output:02conv2d_transpose_59/strided_slice_1/stack:output:04conv2d_transpose_59/strided_slice_1/stack_1:output:04conv2d_transpose_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_59/strided_slice_1?
3conv2d_transpose_59/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_59_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_59/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_59/conv2d_transposeConv2DBackpropInput"conv2d_transpose_59/stack:output:0;conv2d_transpose_59/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$conv2d_transpose_59/conv2d_transpose?
*conv2d_transpose_59/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_59/BiasAdd/ReadVariableOp?
conv2d_transpose_59/BiasAddBiasAdd-conv2d_transpose_59/conv2d_transpose:output:02conv2d_transpose_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_59/BiasAdd?
conv2d_transpose_59/ReluRelu$conv2d_transpose_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_59/Relu?
conv2d_transpose_60/ShapeShape&conv2d_transpose_59/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_60/Shape?
'conv2d_transpose_60/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_60/strided_slice/stack?
)conv2d_transpose_60/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_60/strided_slice/stack_1?
)conv2d_transpose_60/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_60/strided_slice/stack_2?
!conv2d_transpose_60/strided_sliceStridedSlice"conv2d_transpose_60/Shape:output:00conv2d_transpose_60/strided_slice/stack:output:02conv2d_transpose_60/strided_slice/stack_1:output:02conv2d_transpose_60/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_60/strided_slice|
conv2d_transpose_60/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_60/stack/1|
conv2d_transpose_60/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_60/stack/2|
conv2d_transpose_60/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_60/stack/3?
conv2d_transpose_60/stackPack*conv2d_transpose_60/strided_slice:output:0$conv2d_transpose_60/stack/1:output:0$conv2d_transpose_60/stack/2:output:0$conv2d_transpose_60/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_60/stack?
)conv2d_transpose_60/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_60/strided_slice_1/stack?
+conv2d_transpose_60/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_60/strided_slice_1/stack_1?
+conv2d_transpose_60/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_60/strided_slice_1/stack_2?
#conv2d_transpose_60/strided_slice_1StridedSlice"conv2d_transpose_60/stack:output:02conv2d_transpose_60/strided_slice_1/stack:output:04conv2d_transpose_60/strided_slice_1/stack_1:output:04conv2d_transpose_60/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_60/strided_slice_1?
3conv2d_transpose_60/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_60_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_60/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_60/conv2d_transposeConv2DBackpropInput"conv2d_transpose_60/stack:output:0;conv2d_transpose_60/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_59/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2&
$conv2d_transpose_60/conv2d_transpose?
*conv2d_transpose_60/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_60/BiasAdd/ReadVariableOp?
conv2d_transpose_60/BiasAddBiasAdd-conv2d_transpose_60/conv2d_transpose:output:02conv2d_transpose_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_60/BiasAdd?
conv2d_transpose_60/ReluRelu$conv2d_transpose_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_60/Relu?
conv2d_transpose_61/ShapeShape&conv2d_transpose_60/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_61/Shape?
'conv2d_transpose_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_61/strided_slice/stack?
)conv2d_transpose_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_61/strided_slice/stack_1?
)conv2d_transpose_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_61/strided_slice/stack_2?
!conv2d_transpose_61/strided_sliceStridedSlice"conv2d_transpose_61/Shape:output:00conv2d_transpose_61/strided_slice/stack:output:02conv2d_transpose_61/strided_slice/stack_1:output:02conv2d_transpose_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_61/strided_slice|
conv2d_transpose_61/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_61/stack/1|
conv2d_transpose_61/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_61/stack/2|
conv2d_transpose_61/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_61/stack/3?
conv2d_transpose_61/stackPack*conv2d_transpose_61/strided_slice:output:0$conv2d_transpose_61/stack/1:output:0$conv2d_transpose_61/stack/2:output:0$conv2d_transpose_61/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_61/stack?
)conv2d_transpose_61/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_61/strided_slice_1/stack?
+conv2d_transpose_61/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_61/strided_slice_1/stack_1?
+conv2d_transpose_61/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_61/strided_slice_1/stack_2?
#conv2d_transpose_61/strided_slice_1StridedSlice"conv2d_transpose_61/stack:output:02conv2d_transpose_61/strided_slice_1/stack:output:04conv2d_transpose_61/strided_slice_1/stack_1:output:04conv2d_transpose_61/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_61/strided_slice_1?
3conv2d_transpose_61/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_61_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_61/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_61/conv2d_transposeConv2DBackpropInput"conv2d_transpose_61/stack:output:0;conv2d_transpose_61/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_60/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2&
$conv2d_transpose_61/conv2d_transpose?
*conv2d_transpose_61/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_61/BiasAdd/ReadVariableOp?
conv2d_transpose_61/BiasAddBiasAdd-conv2d_transpose_61/conv2d_transpose:output:02conv2d_transpose_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_61/BiasAdd?
IdentityIdentity$conv2d_transpose_61/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp7^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1+^conv2d_transpose_57/BiasAdd/ReadVariableOp4^conv2d_transpose_57/conv2d_transpose/ReadVariableOp+^conv2d_transpose_58/BiasAdd/ReadVariableOp4^conv2d_transpose_58/conv2d_transpose/ReadVariableOp+^conv2d_transpose_59/BiasAdd/ReadVariableOp4^conv2d_transpose_59/conv2d_transpose/ReadVariableOp+^conv2d_transpose_60/BiasAdd/ReadVariableOp4^conv2d_transpose_60/conv2d_transpose/ReadVariableOp+^conv2d_transpose_61/BiasAdd/ReadVariableOp4^conv2d_transpose_61/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 2p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12X
*conv2d_transpose_57/BiasAdd/ReadVariableOp*conv2d_transpose_57/BiasAdd/ReadVariableOp2j
3conv2d_transpose_57/conv2d_transpose/ReadVariableOp3conv2d_transpose_57/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_58/BiasAdd/ReadVariableOp*conv2d_transpose_58/BiasAdd/ReadVariableOp2j
3conv2d_transpose_58/conv2d_transpose/ReadVariableOp3conv2d_transpose_58/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_59/BiasAdd/ReadVariableOp*conv2d_transpose_59/BiasAdd/ReadVariableOp2j
3conv2d_transpose_59/conv2d_transpose/ReadVariableOp3conv2d_transpose_59/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_60/BiasAdd/ReadVariableOp*conv2d_transpose_60/BiasAdd/ReadVariableOp2j
3conv2d_transpose_60/conv2d_transpose/ReadVariableOp3conv2d_transpose_60/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_61/BiasAdd/ReadVariableOp*conv2d_transpose_61/BiasAdd/ReadVariableOp2j
3conv2d_transpose_61/conv2d_transpose/ReadVariableOp3conv2d_transpose_61/conv2d_transpose/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_57_layer_call_fn_5681510

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_56790262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_33_layer_call_fn_5681696

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_56792662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?%
?
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5678931

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681240

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
??
?
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680431

inputsD
6encoder_batch_normalization_31_readvariableop_resource:F
8encoder_batch_normalization_31_readvariableop_1_resource:U
Gencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_resource:W
Iencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource:J
0encoder_conv2d_60_conv2d_readvariableop_resource:?
1encoder_conv2d_60_biasadd_readvariableop_resource:J
0encoder_conv2d_61_conv2d_readvariableop_resource:?
1encoder_conv2d_61_biasadd_readvariableop_resource:D
6encoder_batch_normalization_32_readvariableop_resource:F
8encoder_batch_normalization_32_readvariableop_1_resource:U
Gencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:W
Iencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:J
0encoder_conv2d_62_conv2d_readvariableop_resource:?
1encoder_conv2d_62_biasadd_readvariableop_resource:J
0encoder_conv2d_63_conv2d_readvariableop_resource:?
1encoder_conv2d_63_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_57_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_57_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_58_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_58_biasadd_readvariableop_resource:D
6decoder_batch_normalization_33_readvariableop_resource:F
8decoder_batch_normalization_33_readvariableop_1_resource:U
Gdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:W
Idecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:^
Ddecoder_conv2d_transpose_59_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_59_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_60_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_60_biasadd_readvariableop_resource:^
Ddecoder_conv2d_transpose_61_conv2d_transpose_readvariableop_resource:I
;decoder_conv2d_transpose_61_biasadd_readvariableop_resource:
identity??>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?-decoder/batch_normalization_33/ReadVariableOp?/decoder/batch_normalization_33/ReadVariableOp_1?2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp?2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp?;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp?>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?-encoder/batch_normalization_31/ReadVariableOp?/encoder/batch_normalization_31/ReadVariableOp_1?>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?-encoder/batch_normalization_32/ReadVariableOp?/encoder/batch_normalization_32/ReadVariableOp_1?(encoder/conv2d_60/BiasAdd/ReadVariableOp?'encoder/conv2d_60/Conv2D/ReadVariableOp?(encoder/conv2d_61/BiasAdd/ReadVariableOp?'encoder/conv2d_61/Conv2D/ReadVariableOp?(encoder/conv2d_62/BiasAdd/ReadVariableOp?'encoder/conv2d_62/Conv2D/ReadVariableOp?(encoder/conv2d_63/BiasAdd/ReadVariableOp?'encoder/conv2d_63/Conv2D/ReadVariableOp?
-encoder/batch_normalization_31/ReadVariableOpReadVariableOp6encoder_batch_normalization_31_readvariableop_resource*
_output_shapes
:*
dtype02/
-encoder/batch_normalization_31/ReadVariableOp?
/encoder/batch_normalization_31/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_31_readvariableop_1_resource*
_output_shapes
:*
dtype021
/encoder/batch_normalization_31/ReadVariableOp_1?
>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp?
@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_31_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1?
/encoder/batch_normalization_31/FusedBatchNormV3FusedBatchNormV3inputs5encoder/batch_normalization_31/ReadVariableOp:value:07encoder/batch_normalization_31/ReadVariableOp_1:value:0Fencoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp:value:0Hencoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@:::::*
epsilon%o?:*
is_training( 21
/encoder/batch_normalization_31/FusedBatchNormV3?
'encoder/conv2d_60/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_60/Conv2D/ReadVariableOp?
encoder/conv2d_60/Conv2DConv2D3encoder/batch_normalization_31/FusedBatchNormV3:y:0/encoder/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
encoder/conv2d_60/Conv2D?
(encoder/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_60/BiasAdd/ReadVariableOp?
encoder/conv2d_60/BiasAddBiasAdd!encoder/conv2d_60/Conv2D:output:00encoder/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
encoder/conv2d_60/BiasAdd?
encoder/conv2d_60/ReluRelu"encoder/conv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
encoder/conv2d_60/Relu?
'encoder/conv2d_61/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_61_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_61/Conv2D/ReadVariableOp?
encoder/conv2d_61/Conv2DConv2D$encoder/conv2d_60/Relu:activations:0/encoder/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
encoder/conv2d_61/Conv2D?
(encoder/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_61/BiasAdd/ReadVariableOp?
encoder/conv2d_61/BiasAddBiasAdd!encoder/conv2d_61/Conv2D:output:00encoder/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_61/BiasAdd?
encoder/conv2d_61/ReluRelu"encoder/conv2d_61/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_61/Relu?
-encoder/batch_normalization_32/ReadVariableOpReadVariableOp6encoder_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype02/
-encoder/batch_normalization_32/ReadVariableOp?
/encoder/batch_normalization_32/ReadVariableOp_1ReadVariableOp8encoder_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype021
/encoder/batch_normalization_32/ReadVariableOp_1?
>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpGencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp?
@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIencoder_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1?
/encoder/batch_normalization_32/FusedBatchNormV3FusedBatchNormV3$encoder/conv2d_61/Relu:activations:05encoder/batch_normalization_32/ReadVariableOp:value:07encoder/batch_normalization_32/ReadVariableOp_1:value:0Fencoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Hencoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 21
/encoder/batch_normalization_32/FusedBatchNormV3?
'encoder/conv2d_62/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_62/Conv2D/ReadVariableOp?
encoder/conv2d_62/Conv2DConv2D3encoder/batch_normalization_32/FusedBatchNormV3:y:0/encoder/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
encoder/conv2d_62/Conv2D?
(encoder/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_62_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_62/BiasAdd/ReadVariableOp?
encoder/conv2d_62/BiasAddBiasAdd!encoder/conv2d_62/Conv2D:output:00encoder/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_62/BiasAdd?
encoder/conv2d_62/ReluRelu"encoder/conv2d_62/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
encoder/conv2d_62/Relu?
'encoder/conv2d_63/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'encoder/conv2d_63/Conv2D/ReadVariableOp?
encoder/conv2d_63/Conv2DConv2D$encoder/conv2d_62/Relu:activations:0/encoder/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder/conv2d_63/Conv2D?
(encoder/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(encoder/conv2d_63/BiasAdd/ReadVariableOp?
encoder/conv2d_63/BiasAddBiasAdd!encoder/conv2d_63/Conv2D:output:00encoder/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d_63/BiasAdd?
encoder/conv2d_63/ReluRelu"encoder/conv2d_63/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d_63/Relu?
encoder/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
encoder/flatten_8/Const?
encoder/flatten_8/ReshapeReshape$encoder/conv2d_63/Relu:activations:0 encoder/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten_8/Reshape?
decoder/reshape_8/ShapeShape"encoder/flatten_8/Reshape:output:0*
T0*
_output_shapes
:2
decoder/reshape_8/Shape?
%decoder/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%decoder/reshape_8/strided_slice/stack?
'decoder/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_8/strided_slice/stack_1?
'decoder/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'decoder/reshape_8/strided_slice/stack_2?
decoder/reshape_8/strided_sliceStridedSlice decoder/reshape_8/Shape:output:0.decoder/reshape_8/strided_slice/stack:output:00decoder/reshape_8/strided_slice/stack_1:output:00decoder/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
decoder/reshape_8/strided_slice?
!decoder/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_8/Reshape/shape/1?
!decoder/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_8/Reshape/shape/2?
!decoder/reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!decoder/reshape_8/Reshape/shape/3?
decoder/reshape_8/Reshape/shapePack(decoder/reshape_8/strided_slice:output:0*decoder/reshape_8/Reshape/shape/1:output:0*decoder/reshape_8/Reshape/shape/2:output:0*decoder/reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
decoder/reshape_8/Reshape/shape?
decoder/reshape_8/ReshapeReshape"encoder/flatten_8/Reshape:output:0(decoder/reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
decoder/reshape_8/Reshape?
!decoder/conv2d_transpose_57/ShapeShape"decoder/reshape_8/Reshape:output:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_57/Shape?
/decoder/conv2d_transpose_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_57/strided_slice/stack?
1decoder/conv2d_transpose_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_57/strided_slice/stack_1?
1decoder/conv2d_transpose_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_57/strided_slice/stack_2?
)decoder/conv2d_transpose_57/strided_sliceStridedSlice*decoder/conv2d_transpose_57/Shape:output:08decoder/conv2d_transpose_57/strided_slice/stack:output:0:decoder/conv2d_transpose_57/strided_slice/stack_1:output:0:decoder/conv2d_transpose_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_57/strided_slice?
#decoder/conv2d_transpose_57/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_57/stack/1?
#decoder/conv2d_transpose_57/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_57/stack/2?
#decoder/conv2d_transpose_57/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_57/stack/3?
!decoder/conv2d_transpose_57/stackPack2decoder/conv2d_transpose_57/strided_slice:output:0,decoder/conv2d_transpose_57/stack/1:output:0,decoder/conv2d_transpose_57/stack/2:output:0,decoder/conv2d_transpose_57/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_57/stack?
1decoder/conv2d_transpose_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_57/strided_slice_1/stack?
3decoder/conv2d_transpose_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_57/strided_slice_1/stack_1?
3decoder/conv2d_transpose_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_57/strided_slice_1/stack_2?
+decoder/conv2d_transpose_57/strided_slice_1StridedSlice*decoder/conv2d_transpose_57/stack:output:0:decoder/conv2d_transpose_57/strided_slice_1/stack:output:0<decoder/conv2d_transpose_57/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_57/strided_slice_1?
;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_57_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_57/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_57/stack:output:0Cdecoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp:value:0"decoder/reshape_8/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2.
,decoder/conv2d_transpose_57/conv2d_transpose?
2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_57/BiasAddBiasAdd5decoder/conv2d_transpose_57/conv2d_transpose:output:0:decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2%
#decoder/conv2d_transpose_57/BiasAdd?
 decoder/conv2d_transpose_57/ReluRelu,decoder/conv2d_transpose_57/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 decoder/conv2d_transpose_57/Relu?
!decoder/conv2d_transpose_58/ShapeShape.decoder/conv2d_transpose_57/Relu:activations:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_58/Shape?
/decoder/conv2d_transpose_58/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_58/strided_slice/stack?
1decoder/conv2d_transpose_58/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_58/strided_slice/stack_1?
1decoder/conv2d_transpose_58/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_58/strided_slice/stack_2?
)decoder/conv2d_transpose_58/strided_sliceStridedSlice*decoder/conv2d_transpose_58/Shape:output:08decoder/conv2d_transpose_58/strided_slice/stack:output:0:decoder/conv2d_transpose_58/strided_slice/stack_1:output:0:decoder/conv2d_transpose_58/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_58/strided_slice?
#decoder/conv2d_transpose_58/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_58/stack/1?
#decoder/conv2d_transpose_58/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_58/stack/2?
#decoder/conv2d_transpose_58/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_58/stack/3?
!decoder/conv2d_transpose_58/stackPack2decoder/conv2d_transpose_58/strided_slice:output:0,decoder/conv2d_transpose_58/stack/1:output:0,decoder/conv2d_transpose_58/stack/2:output:0,decoder/conv2d_transpose_58/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_58/stack?
1decoder/conv2d_transpose_58/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_58/strided_slice_1/stack?
3decoder/conv2d_transpose_58/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_58/strided_slice_1/stack_1?
3decoder/conv2d_transpose_58/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_58/strided_slice_1/stack_2?
+decoder/conv2d_transpose_58/strided_slice_1StridedSlice*decoder/conv2d_transpose_58/stack:output:0:decoder/conv2d_transpose_58/strided_slice_1/stack:output:0<decoder/conv2d_transpose_58/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_58/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_58/strided_slice_1?
;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_58_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_58/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_58/stack:output:0Cdecoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_57/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2.
,decoder/conv2d_transpose_58/conv2d_transpose?
2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_58/BiasAddBiasAdd5decoder/conv2d_transpose_58/conv2d_transpose:output:0:decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2%
#decoder/conv2d_transpose_58/BiasAdd?
 decoder/conv2d_transpose_58/ReluRelu,decoder/conv2d_transpose_58/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2"
 decoder/conv2d_transpose_58/Relu?
-decoder/batch_normalization_33/ReadVariableOpReadVariableOp6decoder_batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype02/
-decoder/batch_normalization_33/ReadVariableOp?
/decoder/batch_normalization_33/ReadVariableOp_1ReadVariableOp8decoder_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype021
/decoder/batch_normalization_33/ReadVariableOp_1?
>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpGdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIdecoder_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
/decoder/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3.decoder/conv2d_transpose_58/Relu:activations:05decoder/batch_normalization_33/ReadVariableOp:value:07decoder/batch_normalization_33/ReadVariableOp_1:value:0Fdecoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Hdecoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 21
/decoder/batch_normalization_33/FusedBatchNormV3?
!decoder/conv2d_transpose_59/ShapeShape3decoder/batch_normalization_33/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_59/Shape?
/decoder/conv2d_transpose_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_59/strided_slice/stack?
1decoder/conv2d_transpose_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_59/strided_slice/stack_1?
1decoder/conv2d_transpose_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_59/strided_slice/stack_2?
)decoder/conv2d_transpose_59/strided_sliceStridedSlice*decoder/conv2d_transpose_59/Shape:output:08decoder/conv2d_transpose_59/strided_slice/stack:output:0:decoder/conv2d_transpose_59/strided_slice/stack_1:output:0:decoder/conv2d_transpose_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_59/strided_slice?
#decoder/conv2d_transpose_59/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_59/stack/1?
#decoder/conv2d_transpose_59/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2%
#decoder/conv2d_transpose_59/stack/2?
#decoder/conv2d_transpose_59/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_59/stack/3?
!decoder/conv2d_transpose_59/stackPack2decoder/conv2d_transpose_59/strided_slice:output:0,decoder/conv2d_transpose_59/stack/1:output:0,decoder/conv2d_transpose_59/stack/2:output:0,decoder/conv2d_transpose_59/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_59/stack?
1decoder/conv2d_transpose_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_59/strided_slice_1/stack?
3decoder/conv2d_transpose_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_59/strided_slice_1/stack_1?
3decoder/conv2d_transpose_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_59/strided_slice_1/stack_2?
+decoder/conv2d_transpose_59/strided_slice_1StridedSlice*decoder/conv2d_transpose_59/stack:output:0:decoder/conv2d_transpose_59/strided_slice_1/stack:output:0<decoder/conv2d_transpose_59/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_59/strided_slice_1?
;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_59_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_59/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_59/stack:output:0Cdecoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp:value:03decoder/batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2.
,decoder/conv2d_transpose_59/conv2d_transpose?
2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_59/BiasAddBiasAdd5decoder/conv2d_transpose_59/conv2d_transpose:output:0:decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2%
#decoder/conv2d_transpose_59/BiasAdd?
 decoder/conv2d_transpose_59/ReluRelu,decoder/conv2d_transpose_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2"
 decoder/conv2d_transpose_59/Relu?
!decoder/conv2d_transpose_60/ShapeShape.decoder/conv2d_transpose_59/Relu:activations:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_60/Shape?
/decoder/conv2d_transpose_60/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_60/strided_slice/stack?
1decoder/conv2d_transpose_60/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_60/strided_slice/stack_1?
1decoder/conv2d_transpose_60/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_60/strided_slice/stack_2?
)decoder/conv2d_transpose_60/strided_sliceStridedSlice*decoder/conv2d_transpose_60/Shape:output:08decoder/conv2d_transpose_60/strided_slice/stack:output:0:decoder/conv2d_transpose_60/strided_slice/stack_1:output:0:decoder/conv2d_transpose_60/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_60/strided_slice?
#decoder/conv2d_transpose_60/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_60/stack/1?
#decoder/conv2d_transpose_60/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_60/stack/2?
#decoder/conv2d_transpose_60/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_60/stack/3?
!decoder/conv2d_transpose_60/stackPack2decoder/conv2d_transpose_60/strided_slice:output:0,decoder/conv2d_transpose_60/stack/1:output:0,decoder/conv2d_transpose_60/stack/2:output:0,decoder/conv2d_transpose_60/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_60/stack?
1decoder/conv2d_transpose_60/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_60/strided_slice_1/stack?
3decoder/conv2d_transpose_60/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_60/strided_slice_1/stack_1?
3decoder/conv2d_transpose_60/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_60/strided_slice_1/stack_2?
+decoder/conv2d_transpose_60/strided_slice_1StridedSlice*decoder/conv2d_transpose_60/stack:output:0:decoder/conv2d_transpose_60/strided_slice_1/stack:output:0<decoder/conv2d_transpose_60/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_60/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_60/strided_slice_1?
;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_60_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_60/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_60/stack:output:0Cdecoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_59/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2.
,decoder/conv2d_transpose_60/conv2d_transpose?
2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_60/BiasAddBiasAdd5decoder/conv2d_transpose_60/conv2d_transpose:output:0:decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2%
#decoder/conv2d_transpose_60/BiasAdd?
 decoder/conv2d_transpose_60/ReluRelu,decoder/conv2d_transpose_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2"
 decoder/conv2d_transpose_60/Relu?
!decoder/conv2d_transpose_61/ShapeShape.decoder/conv2d_transpose_60/Relu:activations:0*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_61/Shape?
/decoder/conv2d_transpose_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/decoder/conv2d_transpose_61/strided_slice/stack?
1decoder/conv2d_transpose_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_61/strided_slice/stack_1?
1decoder/conv2d_transpose_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1decoder/conv2d_transpose_61/strided_slice/stack_2?
)decoder/conv2d_transpose_61/strided_sliceStridedSlice*decoder/conv2d_transpose_61/Shape:output:08decoder/conv2d_transpose_61/strided_slice/stack:output:0:decoder/conv2d_transpose_61/strided_slice/stack_1:output:0:decoder/conv2d_transpose_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)decoder/conv2d_transpose_61/strided_slice?
#decoder/conv2d_transpose_61/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_61/stack/1?
#decoder/conv2d_transpose_61/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2%
#decoder/conv2d_transpose_61/stack/2?
#decoder/conv2d_transpose_61/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#decoder/conv2d_transpose_61/stack/3?
!decoder/conv2d_transpose_61/stackPack2decoder/conv2d_transpose_61/strided_slice:output:0,decoder/conv2d_transpose_61/stack/1:output:0,decoder/conv2d_transpose_61/stack/2:output:0,decoder/conv2d_transpose_61/stack/3:output:0*
N*
T0*
_output_shapes
:2#
!decoder/conv2d_transpose_61/stack?
1decoder/conv2d_transpose_61/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1decoder/conv2d_transpose_61/strided_slice_1/stack?
3decoder/conv2d_transpose_61/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_61/strided_slice_1/stack_1?
3decoder/conv2d_transpose_61/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3decoder/conv2d_transpose_61/strided_slice_1/stack_2?
+decoder/conv2d_transpose_61/strided_slice_1StridedSlice*decoder/conv2d_transpose_61/stack:output:0:decoder/conv2d_transpose_61/strided_slice_1/stack:output:0<decoder/conv2d_transpose_61/strided_slice_1/stack_1:output:0<decoder/conv2d_transpose_61/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+decoder/conv2d_transpose_61/strided_slice_1?
;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOpReadVariableOpDdecoder_conv2d_transpose_61_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp?
,decoder/conv2d_transpose_61/conv2d_transposeConv2DBackpropInput*decoder/conv2d_transpose_61/stack:output:0Cdecoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp:value:0.decoder/conv2d_transpose_60/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2.
,decoder/conv2d_transpose_61/conv2d_transpose?
2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOpReadVariableOp;decoder_conv2d_transpose_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp?
#decoder/conv2d_transpose_61/BiasAddBiasAdd5decoder/conv2d_transpose_61/conv2d_transpose:output:0:decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2%
#decoder/conv2d_transpose_61/BiasAdd?
IdentityIdentity,decoder/conv2d_transpose_61/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp?^decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOpA^decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1.^decoder/batch_normalization_33/ReadVariableOp0^decoder/batch_normalization_33/ReadVariableOp_13^decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp3^decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp<^decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp?^encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOpA^encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1.^encoder/batch_normalization_31/ReadVariableOp0^encoder/batch_normalization_31/ReadVariableOp_1?^encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOpA^encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1.^encoder/batch_normalization_32/ReadVariableOp0^encoder/batch_normalization_32/ReadVariableOp_1)^encoder/conv2d_60/BiasAdd/ReadVariableOp(^encoder/conv2d_60/Conv2D/ReadVariableOp)^encoder/conv2d_61/BiasAdd/ReadVariableOp(^encoder/conv2d_61/Conv2D/ReadVariableOp)^encoder/conv2d_62/BiasAdd/ReadVariableOp(^encoder/conv2d_62/Conv2D/ReadVariableOp)^encoder/conv2d_63/BiasAdd/ReadVariableOp(^encoder/conv2d_63/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp>decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1@decoder/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12^
-decoder/batch_normalization_33/ReadVariableOp-decoder/batch_normalization_33/ReadVariableOp2b
/decoder/batch_normalization_33/ReadVariableOp_1/decoder/batch_normalization_33/ReadVariableOp_12h
2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_57/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_57/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_58/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_58/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_59/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_59/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_60/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_60/conv2d_transpose/ReadVariableOp2h
2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp2decoder/conv2d_transpose_61/BiasAdd/ReadVariableOp2z
;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp;decoder/conv2d_transpose_61/conv2d_transpose/ReadVariableOp2?
>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp>encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp2?
@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_1@encoder/batch_normalization_31/FusedBatchNormV3/ReadVariableOp_12^
-encoder/batch_normalization_31/ReadVariableOp-encoder/batch_normalization_31/ReadVariableOp2b
/encoder/batch_normalization_31/ReadVariableOp_1/encoder/batch_normalization_31/ReadVariableOp_12?
>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp>encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2?
@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1@encoder/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12^
-encoder/batch_normalization_32/ReadVariableOp-encoder/batch_normalization_32/ReadVariableOp2b
/encoder/batch_normalization_32/ReadVariableOp_1/encoder/batch_normalization_32/ReadVariableOp_12T
(encoder/conv2d_60/BiasAdd/ReadVariableOp(encoder/conv2d_60/BiasAdd/ReadVariableOp2R
'encoder/conv2d_60/Conv2D/ReadVariableOp'encoder/conv2d_60/Conv2D/ReadVariableOp2T
(encoder/conv2d_61/BiasAdd/ReadVariableOp(encoder/conv2d_61/BiasAdd/ReadVariableOp2R
'encoder/conv2d_61/Conv2D/ReadVariableOp'encoder/conv2d_61/Conv2D/ReadVariableOp2T
(encoder/conv2d_62/BiasAdd/ReadVariableOp(encoder/conv2d_62/BiasAdd/ReadVariableOp2R
'encoder/conv2d_62/Conv2D/ReadVariableOp'encoder/conv2d_62/Conv2D/ReadVariableOp2T
(encoder/conv2d_63/BiasAdd/ReadVariableOp(encoder/conv2d_63/BiasAdd/ReadVariableOp2R
'encoder/conv2d_63/Conv2D/ReadVariableOp'encoder/conv2d_63/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_58_layer_call_fn_5681577

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_56785422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681714

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_5680246

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_56797832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5679055

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5677954

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5678101

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5677765

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_63_layer_call_fn_5681451

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_56779922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5681820

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5678756

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_31_layer_call_fn_5681186

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_56781652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????@@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_decoder_layer_call_fn_5679206
placeholder!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56791752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoded audio
?
?
5__inference_conv2d_transpose_58_layer_call_fn_5681586

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_56790552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?.
?
D__inference_decoder_layer_call_and_return_conditional_losses_5679375

inputs5
conv2d_transpose_57_5679340:)
conv2d_transpose_57_5679342:5
conv2d_transpose_58_5679345:)
conv2d_transpose_58_5679347:,
batch_normalization_33_5679350:,
batch_normalization_33_5679352:,
batch_normalization_33_5679354:,
batch_normalization_33_5679356:5
conv2d_transpose_59_5679359:)
conv2d_transpose_59_5679361:5
conv2d_transpose_60_5679364:)
conv2d_transpose_60_5679366:5
conv2d_transpose_61_5679369:)
conv2d_transpose_61_5679371:
identity??.batch_normalization_33/StatefulPartitionedCall?+conv2d_transpose_57/StatefulPartitionedCall?+conv2d_transpose_58/StatefulPartitionedCall?+conv2d_transpose_59/StatefulPartitionedCall?+conv2d_transpose_60/StatefulPartitionedCall?+conv2d_transpose_61/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_56790012
reshape_8/PartitionedCall?
+conv2d_transpose_57/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_transpose_57_5679340conv2d_transpose_57_5679342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_56790262-
+conv2d_transpose_57/StatefulPartitionedCall?
+conv2d_transpose_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_57/StatefulPartitionedCall:output:0conv2d_transpose_58_5679345conv2d_transpose_58_5679347*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_56790552-
+conv2d_transpose_58/StatefulPartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_58/StatefulPartitionedCall:output:0batch_normalization_33_5679350batch_normalization_33_5679352batch_normalization_33_5679354batch_normalization_33_5679356*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_567926620
.batch_normalization_33/StatefulPartitionedCall?
+conv2d_transpose_59/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_transpose_59_5679359conv2d_transpose_59_5679361*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_56791112-
+conv2d_transpose_59/StatefulPartitionedCall?
+conv2d_transpose_60/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_59/StatefulPartitionedCall:output:0conv2d_transpose_60_5679364conv2d_transpose_60_5679366*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_56791402-
+conv2d_transpose_60/StatefulPartitionedCall?
+conv2d_transpose_61/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_60/StatefulPartitionedCall:output:0conv2d_transpose_61_5679369conv2d_transpose_61_5679371*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_56791682-
+conv2d_transpose_61/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_61/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall,^conv2d_transpose_57/StatefulPartitionedCall,^conv2d_transpose_58/StatefulPartitionedCall,^conv2d_transpose_59/StatefulPartitionedCall,^conv2d_transpose_60/StatefulPartitionedCall,^conv2d_transpose_61/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2Z
+conv2d_transpose_57/StatefulPartitionedCall+conv2d_transpose_57/StatefulPartitionedCall2Z
+conv2d_transpose_58/StatefulPartitionedCall+conv2d_transpose_58/StatefulPartitionedCall2Z
+conv2d_transpose_59/StatefulPartitionedCall+conv2d_transpose_59/StatefulPartitionedCall2Z
+conv2d_transpose_60/StatefulPartitionedCall+conv2d_transpose_60/StatefulPartitionedCall2Z
+conv2d_transpose_61/StatefulPartitionedCall+conv2d_transpose_61/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5679078

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?+
?
D__inference_encoder_layer_call_and_return_conditional_losses_5678258

inputs,
batch_normalization_31_5678218:,
batch_normalization_31_5678220:,
batch_normalization_31_5678222:,
batch_normalization_31_5678224:+
conv2d_60_5678227:
conv2d_60_5678229:+
conv2d_61_5678232:
conv2d_61_5678234:,
batch_normalization_32_5678237:,
batch_normalization_32_5678239:,
batch_normalization_32_5678241:,
batch_normalization_32_5678243:+
conv2d_62_5678246:
conv2d_62_5678248:+
conv2d_63_5678251:
conv2d_63_5678253:
identity??.batch_normalization_31/StatefulPartitionedCall?.batch_normalization_32/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?
.batch_normalization_31/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_31_5678218batch_normalization_31_5678220batch_normalization_31_5678222batch_normalization_31_5678224*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_567816520
.batch_normalization_31/StatefulPartitionedCall?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_31/StatefulPartitionedCall:output:0conv2d_60_5678227conv2d_60_5678229*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_56779142#
!conv2d_60/StatefulPartitionedCall?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0conv2d_61_5678232conv2d_61_5678234*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_56779312#
!conv2d_61/StatefulPartitionedCall?
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0batch_normalization_32_5678237batch_normalization_32_5678239batch_normalization_32_5678241batch_normalization_32_5678243*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_567810120
.batch_normalization_32/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_62_5678246conv2d_62_5678248*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_56779752#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_5678251conv2d_63_5678253*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_56779922#
!conv2d_63/StatefulPartitionedCall?
flatten_8/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_56780042
flatten_8/PartitionedCall~
IdentityIdentity"flatten_8/PartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?
NoOpNoOp/^batch_normalization_31/StatefulPartitionedCall/^batch_normalization_32/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 2`
.batch_normalization_31/StatefulPartitionedCall.batch_normalization_31/StatefulPartitionedCall2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_5678042
placeholder
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56780072
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@
(
_user_specified_nameoriginal audio
?
?
8__inference_batch_normalization_32_layer_call_fn_5681350

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_56781012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_5681298

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?.
?
D__inference_decoder_layer_call_and_return_conditional_losses_5679175

inputs5
conv2d_transpose_57_5679027:)
conv2d_transpose_57_5679029:5
conv2d_transpose_58_5679056:)
conv2d_transpose_58_5679058:,
batch_normalization_33_5679079:,
batch_normalization_33_5679081:,
batch_normalization_33_5679083:,
batch_normalization_33_5679085:5
conv2d_transpose_59_5679112:)
conv2d_transpose_59_5679114:5
conv2d_transpose_60_5679141:)
conv2d_transpose_60_5679143:5
conv2d_transpose_61_5679169:)
conv2d_transpose_61_5679171:
identity??.batch_normalization_33/StatefulPartitionedCall?+conv2d_transpose_57/StatefulPartitionedCall?+conv2d_transpose_58/StatefulPartitionedCall?+conv2d_transpose_59/StatefulPartitionedCall?+conv2d_transpose_60/StatefulPartitionedCall?+conv2d_transpose_61/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_56790012
reshape_8/PartitionedCall?
+conv2d_transpose_57/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_transpose_57_5679027conv2d_transpose_57_5679029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_56790262-
+conv2d_transpose_57/StatefulPartitionedCall?
+conv2d_transpose_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_57/StatefulPartitionedCall:output:0conv2d_transpose_58_5679056conv2d_transpose_58_5679058*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_56790552-
+conv2d_transpose_58/StatefulPartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_58/StatefulPartitionedCall:output:0batch_normalization_33_5679079batch_normalization_33_5679081batch_normalization_33_5679083batch_normalization_33_5679085*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_567907820
.batch_normalization_33/StatefulPartitionedCall?
+conv2d_transpose_59/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_transpose_59_5679112conv2d_transpose_59_5679114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_56791112-
+conv2d_transpose_59/StatefulPartitionedCall?
+conv2d_transpose_60/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_59/StatefulPartitionedCall:output:0conv2d_transpose_60_5679141conv2d_transpose_60_5679143*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_56791402-
+conv2d_transpose_60/StatefulPartitionedCall?
+conv2d_transpose_61/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_60/StatefulPartitionedCall:output:0conv2d_transpose_61_5679169conv2d_transpose_61_5679171*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_56791682-
+conv2d_transpose_61/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_61/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall,^conv2d_transpose_57/StatefulPartitionedCall,^conv2d_transpose_58/StatefulPartitionedCall,^conv2d_transpose_59/StatefulPartitionedCall,^conv2d_transpose_60/StatefulPartitionedCall,^conv2d_transpose_61/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2Z
+conv2d_transpose_57/StatefulPartitionedCall+conv2d_transpose_57/StatefulPartitionedCall2Z
+conv2d_transpose_58/StatefulPartitionedCall+conv2d_transpose_58/StatefulPartitionedCall2Z
+conv2d_transpose_59/StatefulPartitionedCall+conv2d_transpose_59/StatefulPartitionedCall2Z
+conv2d_transpose_60/StatefulPartitionedCall+conv2d_transpose_60/StatefulPartitionedCall2Z
+conv2d_transpose_61/StatefulPartitionedCall+conv2d_transpose_61/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_flatten_8_layer_call_fn_5681467

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_flatten_8_layer_call_and_return_conditional_losses_56780042
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
D__inference_decoder_layer_call_and_return_conditional_losses_5681134

inputsV
<conv2d_transpose_57_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_57_biasadd_readvariableop_resource:V
<conv2d_transpose_58_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_58_biasadd_readvariableop_resource:<
.batch_normalization_33_readvariableop_resource:>
0batch_normalization_33_readvariableop_1_resource:M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:V
<conv2d_transpose_59_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_59_biasadd_readvariableop_resource:V
<conv2d_transpose_60_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_60_biasadd_readvariableop_resource:V
<conv2d_transpose_61_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_61_biasadd_readvariableop_resource:
identity??%batch_normalization_33/AssignNewValue?'batch_normalization_33/AssignNewValue_1?6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?*conv2d_transpose_57/BiasAdd/ReadVariableOp?3conv2d_transpose_57/conv2d_transpose/ReadVariableOp?*conv2d_transpose_58/BiasAdd/ReadVariableOp?3conv2d_transpose_58/conv2d_transpose/ReadVariableOp?*conv2d_transpose_59/BiasAdd/ReadVariableOp?3conv2d_transpose_59/conv2d_transpose/ReadVariableOp?*conv2d_transpose_60/BiasAdd/ReadVariableOp?3conv2d_transpose_60/conv2d_transpose/ReadVariableOp?*conv2d_transpose_61/BiasAdd/ReadVariableOp?3conv2d_transpose_61/conv2d_transpose/ReadVariableOpX
reshape_8/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2x
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeinputs reshape_8/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_8/Reshape?
conv2d_transpose_57/ShapeShapereshape_8/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_57/Shape?
'conv2d_transpose_57/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_57/strided_slice/stack?
)conv2d_transpose_57/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_57/strided_slice/stack_1?
)conv2d_transpose_57/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_57/strided_slice/stack_2?
!conv2d_transpose_57/strided_sliceStridedSlice"conv2d_transpose_57/Shape:output:00conv2d_transpose_57/strided_slice/stack:output:02conv2d_transpose_57/strided_slice/stack_1:output:02conv2d_transpose_57/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_57/strided_slice|
conv2d_transpose_57/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_57/stack/1|
conv2d_transpose_57/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_57/stack/2|
conv2d_transpose_57/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_57/stack/3?
conv2d_transpose_57/stackPack*conv2d_transpose_57/strided_slice:output:0$conv2d_transpose_57/stack/1:output:0$conv2d_transpose_57/stack/2:output:0$conv2d_transpose_57/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_57/stack?
)conv2d_transpose_57/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_57/strided_slice_1/stack?
+conv2d_transpose_57/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_57/strided_slice_1/stack_1?
+conv2d_transpose_57/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_57/strided_slice_1/stack_2?
#conv2d_transpose_57/strided_slice_1StridedSlice"conv2d_transpose_57/stack:output:02conv2d_transpose_57/strided_slice_1/stack:output:04conv2d_transpose_57/strided_slice_1/stack_1:output:04conv2d_transpose_57/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_57/strided_slice_1?
3conv2d_transpose_57/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_57_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_57/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_57/conv2d_transposeConv2DBackpropInput"conv2d_transpose_57/stack:output:0;conv2d_transpose_57/conv2d_transpose/ReadVariableOp:value:0reshape_8/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_57/conv2d_transpose?
*conv2d_transpose_57/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_57_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_57/BiasAdd/ReadVariableOp?
conv2d_transpose_57/BiasAddBiasAdd-conv2d_transpose_57/conv2d_transpose:output:02conv2d_transpose_57/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_57/BiasAdd?
conv2d_transpose_57/ReluRelu$conv2d_transpose_57/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_57/Relu?
conv2d_transpose_58/ShapeShape&conv2d_transpose_57/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_58/Shape?
'conv2d_transpose_58/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_58/strided_slice/stack?
)conv2d_transpose_58/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_58/strided_slice/stack_1?
)conv2d_transpose_58/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_58/strided_slice/stack_2?
!conv2d_transpose_58/strided_sliceStridedSlice"conv2d_transpose_58/Shape:output:00conv2d_transpose_58/strided_slice/stack:output:02conv2d_transpose_58/strided_slice/stack_1:output:02conv2d_transpose_58/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_58/strided_slice|
conv2d_transpose_58/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_58/stack/1|
conv2d_transpose_58/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_58/stack/2|
conv2d_transpose_58/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_58/stack/3?
conv2d_transpose_58/stackPack*conv2d_transpose_58/strided_slice:output:0$conv2d_transpose_58/stack/1:output:0$conv2d_transpose_58/stack/2:output:0$conv2d_transpose_58/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_58/stack?
)conv2d_transpose_58/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_58/strided_slice_1/stack?
+conv2d_transpose_58/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_58/strided_slice_1/stack_1?
+conv2d_transpose_58/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_58/strided_slice_1/stack_2?
#conv2d_transpose_58/strided_slice_1StridedSlice"conv2d_transpose_58/stack:output:02conv2d_transpose_58/strided_slice_1/stack:output:04conv2d_transpose_58/strided_slice_1/stack_1:output:04conv2d_transpose_58/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_58/strided_slice_1?
3conv2d_transpose_58/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_58_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_58/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_58/conv2d_transposeConv2DBackpropInput"conv2d_transpose_58/stack:output:0;conv2d_transpose_58/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_57/Relu:activations:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$conv2d_transpose_58/conv2d_transpose?
*conv2d_transpose_58/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_58_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_58/BiasAdd/ReadVariableOp?
conv2d_transpose_58/BiasAddBiasAdd-conv2d_transpose_58/conv2d_transpose:output:02conv2d_transpose_58/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_58/BiasAdd?
conv2d_transpose_58/ReluRelu$conv2d_transpose_58/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_58/Relu?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_33/ReadVariableOp?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_33/ReadVariableOp_1?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3&conv2d_transpose_58/Relu:activations:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2)
'batch_normalization_33/FusedBatchNormV3?
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_33/AssignNewValue?
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02)
'batch_normalization_33/AssignNewValue_1?
conv2d_transpose_59/ShapeShape+batch_normalization_33/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
conv2d_transpose_59/Shape?
'conv2d_transpose_59/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_59/strided_slice/stack?
)conv2d_transpose_59/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_59/strided_slice/stack_1?
)conv2d_transpose_59/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_59/strided_slice/stack_2?
!conv2d_transpose_59/strided_sliceStridedSlice"conv2d_transpose_59/Shape:output:00conv2d_transpose_59/strided_slice/stack:output:02conv2d_transpose_59/strided_slice/stack_1:output:02conv2d_transpose_59/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_59/strided_slice|
conv2d_transpose_59/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_59/stack/1|
conv2d_transpose_59/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_59/stack/2|
conv2d_transpose_59/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_59/stack/3?
conv2d_transpose_59/stackPack*conv2d_transpose_59/strided_slice:output:0$conv2d_transpose_59/stack/1:output:0$conv2d_transpose_59/stack/2:output:0$conv2d_transpose_59/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_59/stack?
)conv2d_transpose_59/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_59/strided_slice_1/stack?
+conv2d_transpose_59/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_59/strided_slice_1/stack_1?
+conv2d_transpose_59/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_59/strided_slice_1/stack_2?
#conv2d_transpose_59/strided_slice_1StridedSlice"conv2d_transpose_59/stack:output:02conv2d_transpose_59/strided_slice_1/stack:output:04conv2d_transpose_59/strided_slice_1/stack_1:output:04conv2d_transpose_59/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_59/strided_slice_1?
3conv2d_transpose_59/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_59_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_59/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_59/conv2d_transposeConv2DBackpropInput"conv2d_transpose_59/stack:output:0;conv2d_transpose_59/conv2d_transpose/ReadVariableOp:value:0+batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2&
$conv2d_transpose_59/conv2d_transpose?
*conv2d_transpose_59/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_59_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_59/BiasAdd/ReadVariableOp?
conv2d_transpose_59/BiasAddBiasAdd-conv2d_transpose_59/conv2d_transpose:output:02conv2d_transpose_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_59/BiasAdd?
conv2d_transpose_59/ReluRelu$conv2d_transpose_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
conv2d_transpose_59/Relu?
conv2d_transpose_60/ShapeShape&conv2d_transpose_59/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_60/Shape?
'conv2d_transpose_60/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_60/strided_slice/stack?
)conv2d_transpose_60/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_60/strided_slice/stack_1?
)conv2d_transpose_60/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_60/strided_slice/stack_2?
!conv2d_transpose_60/strided_sliceStridedSlice"conv2d_transpose_60/Shape:output:00conv2d_transpose_60/strided_slice/stack:output:02conv2d_transpose_60/strided_slice/stack_1:output:02conv2d_transpose_60/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_60/strided_slice|
conv2d_transpose_60/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_60/stack/1|
conv2d_transpose_60/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_60/stack/2|
conv2d_transpose_60/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_60/stack/3?
conv2d_transpose_60/stackPack*conv2d_transpose_60/strided_slice:output:0$conv2d_transpose_60/stack/1:output:0$conv2d_transpose_60/stack/2:output:0$conv2d_transpose_60/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_60/stack?
)conv2d_transpose_60/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_60/strided_slice_1/stack?
+conv2d_transpose_60/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_60/strided_slice_1/stack_1?
+conv2d_transpose_60/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_60/strided_slice_1/stack_2?
#conv2d_transpose_60/strided_slice_1StridedSlice"conv2d_transpose_60/stack:output:02conv2d_transpose_60/strided_slice_1/stack:output:04conv2d_transpose_60/strided_slice_1/stack_1:output:04conv2d_transpose_60/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_60/strided_slice_1?
3conv2d_transpose_60/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_60_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_60/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_60/conv2d_transposeConv2DBackpropInput"conv2d_transpose_60/stack:output:0;conv2d_transpose_60/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_59/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2&
$conv2d_transpose_60/conv2d_transpose?
*conv2d_transpose_60/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_60_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_60/BiasAdd/ReadVariableOp?
conv2d_transpose_60/BiasAddBiasAdd-conv2d_transpose_60/conv2d_transpose:output:02conv2d_transpose_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_60/BiasAdd?
conv2d_transpose_60/ReluRelu$conv2d_transpose_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_60/Relu?
conv2d_transpose_61/ShapeShape&conv2d_transpose_60/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_61/Shape?
'conv2d_transpose_61/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_61/strided_slice/stack?
)conv2d_transpose_61/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_61/strided_slice/stack_1?
)conv2d_transpose_61/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_61/strided_slice/stack_2?
!conv2d_transpose_61/strided_sliceStridedSlice"conv2d_transpose_61/Shape:output:00conv2d_transpose_61/strided_slice/stack:output:02conv2d_transpose_61/strided_slice/stack_1:output:02conv2d_transpose_61/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_61/strided_slice|
conv2d_transpose_61/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_61/stack/1|
conv2d_transpose_61/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_61/stack/2|
conv2d_transpose_61/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_61/stack/3?
conv2d_transpose_61/stackPack*conv2d_transpose_61/strided_slice:output:0$conv2d_transpose_61/stack/1:output:0$conv2d_transpose_61/stack/2:output:0$conv2d_transpose_61/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_61/stack?
)conv2d_transpose_61/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_61/strided_slice_1/stack?
+conv2d_transpose_61/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_61/strided_slice_1/stack_1?
+conv2d_transpose_61/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_61/strided_slice_1/stack_2?
#conv2d_transpose_61/strided_slice_1StridedSlice"conv2d_transpose_61/stack:output:02conv2d_transpose_61/strided_slice_1/stack:output:04conv2d_transpose_61/strided_slice_1/stack_1:output:04conv2d_transpose_61/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_61/strided_slice_1?
3conv2d_transpose_61/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_61_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_61/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_61/conv2d_transposeConv2DBackpropInput"conv2d_transpose_61/stack:output:0;conv2d_transpose_61/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_60/Relu:activations:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2&
$conv2d_transpose_61/conv2d_transpose?
*conv2d_transpose_61/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_61_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_61/BiasAdd/ReadVariableOp?
conv2d_transpose_61/BiasAddBiasAdd-conv2d_transpose_61/conv2d_transpose:output:02conv2d_transpose_61/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_transpose_61/BiasAdd?
IdentityIdentity$conv2d_transpose_61/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1+^conv2d_transpose_57/BiasAdd/ReadVariableOp4^conv2d_transpose_57/conv2d_transpose/ReadVariableOp+^conv2d_transpose_58/BiasAdd/ReadVariableOp4^conv2d_transpose_58/conv2d_transpose/ReadVariableOp+^conv2d_transpose_59/BiasAdd/ReadVariableOp4^conv2d_transpose_59/conv2d_transpose/ReadVariableOp+^conv2d_transpose_60/BiasAdd/ReadVariableOp4^conv2d_transpose_60/conv2d_transpose/ReadVariableOp+^conv2d_transpose_61/BiasAdd/ReadVariableOp4^conv2d_transpose_61/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 2N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12X
*conv2d_transpose_57/BiasAdd/ReadVariableOp*conv2d_transpose_57/BiasAdd/ReadVariableOp2j
3conv2d_transpose_57/conv2d_transpose/ReadVariableOp3conv2d_transpose_57/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_58/BiasAdd/ReadVariableOp*conv2d_transpose_58/BiasAdd/ReadVariableOp2j
3conv2d_transpose_58/conv2d_transpose/ReadVariableOp3conv2d_transpose_58/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_59/BiasAdd/ReadVariableOp*conv2d_transpose_59/BiasAdd/ReadVariableOp2j
3conv2d_transpose_59/conv2d_transpose/ReadVariableOp3conv2d_transpose_59/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_60/BiasAdd/ReadVariableOp*conv2d_transpose_60/BiasAdd/ReadVariableOp2j
3conv2d_transpose_60/conv2d_transpose/ReadVariableOp3conv2d_transpose_60/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_61/BiasAdd/ReadVariableOp*conv2d_transpose_61/BiasAdd/ReadVariableOp2j
3conv2d_transpose_61/conv2d_transpose/ReadVariableOp3conv2d_transpose_61/conv2d_transpose/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_60_layer_call_fn_5681267

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_56779142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681368

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_5678330
placeholder
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56782582
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????@@
(
_user_specified_nameoriginal audio
?%
?
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5681971

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?	
H__inference_autoencoder_layer_call_and_return_conditional_losses_5679587

inputs
encoder_5679524:
encoder_5679526:
encoder_5679528:
encoder_5679530:)
encoder_5679532:
encoder_5679534:)
encoder_5679536:
encoder_5679538:
encoder_5679540:
encoder_5679542:
encoder_5679544:
encoder_5679546:)
encoder_5679548:
encoder_5679550:)
encoder_5679552:
encoder_5679554:)
decoder_5679557:
decoder_5679559:)
decoder_5679561:
decoder_5679563:
decoder_5679565:
decoder_5679567:
decoder_5679569:
decoder_5679571:)
decoder_5679573:
decoder_5679575:)
decoder_5679577:
decoder_5679579:)
decoder_5679581:
decoder_5679583:
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_5679524encoder_5679526encoder_5679528encoder_5679530encoder_5679532encoder_5679534encoder_5679536encoder_5679538encoder_5679540encoder_5679542encoder_5679544encoder_5679546encoder_5679548encoder_5679550encoder_5679552encoder_5679554*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56780072!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_5679557decoder_5679559decoder_5679561decoder_5679563decoder_5679565decoder_5679567decoder_5679569decoder_5679571decoder_5679573decoder_5679575decoder_5679577decoder_5679579decoder_5679581decoder_5679583*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56791752!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5679140

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
)__inference_decoder_layer_call_fn_5680847

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56791752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_decoder_layer_call_fn_5680880

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_56793752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_5681442

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
b
F__inference_flatten_8_layer_call_and_return_conditional_losses_5678004

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_reshape_8_layer_call_fn_5681478

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_56790012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_5681278

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5678658

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_57_layer_call_fn_5681501

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_56784542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681404

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681422

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_59_layer_call_fn_5681777

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_56787562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_batch_normalization_33_layer_call_fn_5681683

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_56790782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_60_layer_call_fn_5681853

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_56788442
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5679168

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_5679650	
input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:$

unknown_17:

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_56795872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*j
_input_shapesY
W:?????????@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
/
_output_shapes
:?????????@@

_user_specified_nameinput
?
b
F__inference_reshape_8_layer_call_and_return_conditional_losses_5681492

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?:
#__inference__traced_restore_5682537
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: =
/assignvariableop_5_batch_normalization_31_gamma:<
.assignvariableop_6_batch_normalization_31_beta:=
#assignvariableop_7_conv2d_60_kernel:/
!assignvariableop_8_conv2d_60_bias:=
#assignvariableop_9_conv2d_61_kernel:0
"assignvariableop_10_conv2d_61_bias:>
0assignvariableop_11_batch_normalization_32_gamma:=
/assignvariableop_12_batch_normalization_32_beta:>
$assignvariableop_13_conv2d_62_kernel:0
"assignvariableop_14_conv2d_62_bias:>
$assignvariableop_15_conv2d_63_kernel:0
"assignvariableop_16_conv2d_63_bias:H
.assignvariableop_17_conv2d_transpose_57_kernel::
,assignvariableop_18_conv2d_transpose_57_bias:H
.assignvariableop_19_conv2d_transpose_58_kernel::
,assignvariableop_20_conv2d_transpose_58_bias:>
0assignvariableop_21_batch_normalization_33_gamma:=
/assignvariableop_22_batch_normalization_33_beta:H
.assignvariableop_23_conv2d_transpose_59_kernel::
,assignvariableop_24_conv2d_transpose_59_bias:H
.assignvariableop_25_conv2d_transpose_60_kernel::
,assignvariableop_26_conv2d_transpose_60_bias:H
.assignvariableop_27_conv2d_transpose_61_kernel::
,assignvariableop_28_conv2d_transpose_61_bias:D
6assignvariableop_29_batch_normalization_31_moving_mean:H
:assignvariableop_30_batch_normalization_31_moving_variance:D
6assignvariableop_31_batch_normalization_32_moving_mean:H
:assignvariableop_32_batch_normalization_32_moving_variance:D
6assignvariableop_33_batch_normalization_33_moving_mean:H
:assignvariableop_34_batch_normalization_33_moving_variance:#
assignvariableop_35_total: #
assignvariableop_36_count: E
7assignvariableop_37_adam_batch_normalization_31_gamma_m:D
6assignvariableop_38_adam_batch_normalization_31_beta_m:E
+assignvariableop_39_adam_conv2d_60_kernel_m:7
)assignvariableop_40_adam_conv2d_60_bias_m:E
+assignvariableop_41_adam_conv2d_61_kernel_m:7
)assignvariableop_42_adam_conv2d_61_bias_m:E
7assignvariableop_43_adam_batch_normalization_32_gamma_m:D
6assignvariableop_44_adam_batch_normalization_32_beta_m:E
+assignvariableop_45_adam_conv2d_62_kernel_m:7
)assignvariableop_46_adam_conv2d_62_bias_m:E
+assignvariableop_47_adam_conv2d_63_kernel_m:7
)assignvariableop_48_adam_conv2d_63_bias_m:O
5assignvariableop_49_adam_conv2d_transpose_57_kernel_m:A
3assignvariableop_50_adam_conv2d_transpose_57_bias_m:O
5assignvariableop_51_adam_conv2d_transpose_58_kernel_m:A
3assignvariableop_52_adam_conv2d_transpose_58_bias_m:E
7assignvariableop_53_adam_batch_normalization_33_gamma_m:D
6assignvariableop_54_adam_batch_normalization_33_beta_m:O
5assignvariableop_55_adam_conv2d_transpose_59_kernel_m:A
3assignvariableop_56_adam_conv2d_transpose_59_bias_m:O
5assignvariableop_57_adam_conv2d_transpose_60_kernel_m:A
3assignvariableop_58_adam_conv2d_transpose_60_bias_m:O
5assignvariableop_59_adam_conv2d_transpose_61_kernel_m:A
3assignvariableop_60_adam_conv2d_transpose_61_bias_m:E
7assignvariableop_61_adam_batch_normalization_31_gamma_v:D
6assignvariableop_62_adam_batch_normalization_31_beta_v:E
+assignvariableop_63_adam_conv2d_60_kernel_v:7
)assignvariableop_64_adam_conv2d_60_bias_v:E
+assignvariableop_65_adam_conv2d_61_kernel_v:7
)assignvariableop_66_adam_conv2d_61_bias_v:E
7assignvariableop_67_adam_batch_normalization_32_gamma_v:D
6assignvariableop_68_adam_batch_normalization_32_beta_v:E
+assignvariableop_69_adam_conv2d_62_kernel_v:7
)assignvariableop_70_adam_conv2d_62_bias_v:E
+assignvariableop_71_adam_conv2d_63_kernel_v:7
)assignvariableop_72_adam_conv2d_63_bias_v:O
5assignvariableop_73_adam_conv2d_transpose_57_kernel_v:A
3assignvariableop_74_adam_conv2d_transpose_57_bias_v:O
5assignvariableop_75_adam_conv2d_transpose_58_kernel_v:A
3assignvariableop_76_adam_conv2d_transpose_58_bias_v:E
7assignvariableop_77_adam_batch_normalization_33_gamma_v:D
6assignvariableop_78_adam_batch_normalization_33_beta_v:O
5assignvariableop_79_adam_conv2d_transpose_59_kernel_v:A
3assignvariableop_80_adam_conv2d_transpose_59_bias_v:O
5assignvariableop_81_adam_conv2d_transpose_60_kernel_v:A
3assignvariableop_82_adam_conv2d_transpose_60_bias_v:O
5assignvariableop_83_adam_conv2d_transpose_61_kernel_v:A
3assignvariableop_84_adam_conv2d_transpose_61_bias_v:
identity_86??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_9?,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*?+
value?+B?+VB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*?
value?B?VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp/assignvariableop_5_batch_normalization_31_gammaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_31_betaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_60_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2d_60_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_61_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_61_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_32_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_32_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_62_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2d_62_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_63_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv2d_63_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_conv2d_transpose_57_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_conv2d_transpose_57_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_conv2d_transpose_58_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_conv2d_transpose_58_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_33_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_33_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_conv2d_transpose_59_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_conv2d_transpose_59_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp.assignvariableop_25_conv2d_transpose_60_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp,assignvariableop_26_conv2d_transpose_60_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_conv2d_transpose_61_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_conv2d_transpose_61_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp6assignvariableop_29_batch_normalization_31_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp:assignvariableop_30_batch_normalization_31_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp6assignvariableop_31_batch_normalization_32_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp:assignvariableop_32_batch_normalization_32_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_batch_normalization_33_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp:assignvariableop_34_batch_normalization_33_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_batch_normalization_31_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_31_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_60_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_60_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_61_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_61_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_batch_normalization_32_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_batch_normalization_32_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_62_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_62_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_63_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_63_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adam_conv2d_transpose_57_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_conv2d_transpose_57_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_conv2d_transpose_58_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_conv2d_transpose_58_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adam_batch_normalization_33_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_33_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_conv2d_transpose_59_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp3assignvariableop_56_adam_conv2d_transpose_59_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp5assignvariableop_57_adam_conv2d_transpose_60_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp3assignvariableop_58_adam_conv2d_transpose_60_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp5assignvariableop_59_adam_conv2d_transpose_61_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp3assignvariableop_60_adam_conv2d_transpose_61_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_batch_normalization_31_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_batch_normalization_31_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_60_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_60_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_61_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_61_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_batch_normalization_32_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp6assignvariableop_68_adam_batch_normalization_32_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_62_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_62_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_63_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_63_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp5assignvariableop_73_adam_conv2d_transpose_57_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp3assignvariableop_74_adam_conv2d_transpose_57_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp5assignvariableop_75_adam_conv2d_transpose_58_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp3assignvariableop_76_adam_conv2d_transpose_58_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_batch_normalization_33_gamma_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adam_batch_normalization_33_beta_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp5assignvariableop_79_adam_conv2d_transpose_59_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp3assignvariableop_80_adam_conv2d_transpose_59_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp5assignvariableop_81_adam_conv2d_transpose_60_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp3assignvariableop_82_adam_conv2d_transpose_60_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp5assignvariableop_83_adam_conv2d_transpose_61_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp3assignvariableop_84_adam_conv2d_transpose_61_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_849
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_85Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_85f
Identity_86IdentityIdentity_85:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_86?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_86Identity_86:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
)__inference_encoder_layer_call_fn_5680653

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56780072
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_5680690

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_56782582
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????@@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_32_layer_call_fn_5681311

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_56777652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5679266

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5678844

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_5677914

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681768

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????  :::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????  : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681222

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?.
?
D__inference_decoder_layer_call_and_return_conditional_losses_5679517
placeholder5
conv2d_transpose_57_5679482:)
conv2d_transpose_57_5679484:5
conv2d_transpose_58_5679487:)
conv2d_transpose_58_5679489:,
batch_normalization_33_5679492:,
batch_normalization_33_5679494:,
batch_normalization_33_5679496:,
batch_normalization_33_5679498:5
conv2d_transpose_59_5679501:)
conv2d_transpose_59_5679503:5
conv2d_transpose_60_5679506:)
conv2d_transpose_60_5679508:5
conv2d_transpose_61_5679511:)
conv2d_transpose_61_5679513:
identity??.batch_normalization_33/StatefulPartitionedCall?+conv2d_transpose_57/StatefulPartitionedCall?+conv2d_transpose_58/StatefulPartitionedCall?+conv2d_transpose_59/StatefulPartitionedCall?+conv2d_transpose_60/StatefulPartitionedCall?+conv2d_transpose_61/StatefulPartitionedCall?
reshape_8/PartitionedCallPartitionedCallplaceholder*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_reshape_8_layer_call_and_return_conditional_losses_56790012
reshape_8/PartitionedCall?
+conv2d_transpose_57/StatefulPartitionedCallStatefulPartitionedCall"reshape_8/PartitionedCall:output:0conv2d_transpose_57_5679482conv2d_transpose_57_5679484*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_56790262-
+conv2d_transpose_57/StatefulPartitionedCall?
+conv2d_transpose_58/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_57/StatefulPartitionedCall:output:0conv2d_transpose_58_5679487conv2d_transpose_58_5679489*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_56790552-
+conv2d_transpose_58/StatefulPartitionedCall?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_58/StatefulPartitionedCall:output:0batch_normalization_33_5679492batch_normalization_33_5679494batch_normalization_33_5679496batch_normalization_33_5679498*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_567926620
.batch_normalization_33/StatefulPartitionedCall?
+conv2d_transpose_59/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_transpose_59_5679501conv2d_transpose_59_5679503*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_56791112-
+conv2d_transpose_59/StatefulPartitionedCall?
+conv2d_transpose_60/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_59/StatefulPartitionedCall:output:0conv2d_transpose_60_5679506conv2d_transpose_60_5679508*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_56791402-
+conv2d_transpose_60/StatefulPartitionedCall?
+conv2d_transpose_61/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_60/StatefulPartitionedCall:output:0conv2d_transpose_61_5679511conv2d_transpose_61_5679513*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_56791682-
+conv2d_transpose_61/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_61/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identity?
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall,^conv2d_transpose_57/StatefulPartitionedCall,^conv2d_transpose_58/StatefulPartitionedCall,^conv2d_transpose_59/StatefulPartitionedCall,^conv2d_transpose_60/StatefulPartitionedCall,^conv2d_transpose_61/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2Z
+conv2d_transpose_57/StatefulPartitionedCall+conv2d_transpose_57/StatefulPartitionedCall2Z
+conv2d_transpose_58/StatefulPartitionedCall+conv2d_transpose_58/StatefulPartitionedCall2Z
+conv2d_transpose_59/StatefulPartitionedCall+conv2d_transpose_59/StatefulPartitionedCall2Z
+conv2d_transpose_60/StatefulPartitionedCall+conv2d_transpose_60/StatefulPartitionedCall2Z
+conv2d_transpose_61/StatefulPartitionedCall+conv2d_transpose_61/StatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_nameencoded audio
?
?
5__inference_conv2d_transpose_59_layer_call_fn_5681786

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_56791112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5681644

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:?????????  *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????  2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????  2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5681620

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_60_layer_call_fn_5681862

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_56791402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_5677992

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
??
?(
 __inference__traced_save_5682272
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop;
7savev2_batch_normalization_31_gamma_read_readvariableop:
6savev2_batch_normalization_31_beta_read_readvariableop/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop/
+savev2_conv2d_61_kernel_read_readvariableop-
)savev2_conv2d_61_bias_read_readvariableop;
7savev2_batch_normalization_32_gamma_read_readvariableop:
6savev2_batch_normalization_32_beta_read_readvariableop/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop9
5savev2_conv2d_transpose_57_kernel_read_readvariableop7
3savev2_conv2d_transpose_57_bias_read_readvariableop9
5savev2_conv2d_transpose_58_kernel_read_readvariableop7
3savev2_conv2d_transpose_58_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableop9
5savev2_conv2d_transpose_59_kernel_read_readvariableop7
3savev2_conv2d_transpose_59_bias_read_readvariableop9
5savev2_conv2d_transpose_60_kernel_read_readvariableop7
3savev2_conv2d_transpose_60_bias_read_readvariableop9
5savev2_conv2d_transpose_61_kernel_read_readvariableop7
3savev2_conv2d_transpose_61_bias_read_readvariableopA
=savev2_batch_normalization_31_moving_mean_read_readvariableopE
Asavev2_batch_normalization_31_moving_variance_read_readvariableopA
=savev2_batch_normalization_32_moving_mean_read_readvariableopE
Asavev2_batch_normalization_32_moving_variance_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopB
>savev2_adam_batch_normalization_31_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_31_beta_m_read_readvariableop6
2savev2_adam_conv2d_60_kernel_m_read_readvariableop4
0savev2_adam_conv2d_60_bias_m_read_readvariableop6
2savev2_adam_conv2d_61_kernel_m_read_readvariableop4
0savev2_adam_conv2d_61_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_m_read_readvariableop6
2savev2_adam_conv2d_62_kernel_m_read_readvariableop4
0savev2_adam_conv2d_62_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_57_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_57_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_58_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_58_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_59_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_59_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_60_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_60_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_61_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_61_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_31_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_31_beta_v_read_readvariableop6
2savev2_adam_conv2d_60_kernel_v_read_readvariableop4
0savev2_adam_conv2d_60_bias_v_read_readvariableop6
2savev2_adam_conv2d_61_kernel_v_read_readvariableop4
0savev2_adam_conv2d_61_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_32_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_32_beta_v_read_readvariableop6
2savev2_adam_conv2d_62_kernel_v_read_readvariableop4
0savev2_adam_conv2d_62_bias_v_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_57_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_57_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_58_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_58_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_59_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_59_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_60_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_60_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_61_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_61_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*?+
value?+B?+VB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:V*
dtype0*?
value?B?VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?&
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop7savev2_batch_normalization_31_gamma_read_readvariableop6savev2_batch_normalization_31_beta_read_readvariableop+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop+savev2_conv2d_61_kernel_read_readvariableop)savev2_conv2d_61_bias_read_readvariableop7savev2_batch_normalization_32_gamma_read_readvariableop6savev2_batch_normalization_32_beta_read_readvariableop+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop5savev2_conv2d_transpose_57_kernel_read_readvariableop3savev2_conv2d_transpose_57_bias_read_readvariableop5savev2_conv2d_transpose_58_kernel_read_readvariableop3savev2_conv2d_transpose_58_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop5savev2_conv2d_transpose_59_kernel_read_readvariableop3savev2_conv2d_transpose_59_bias_read_readvariableop5savev2_conv2d_transpose_60_kernel_read_readvariableop3savev2_conv2d_transpose_60_bias_read_readvariableop5savev2_conv2d_transpose_61_kernel_read_readvariableop3savev2_conv2d_transpose_61_bias_read_readvariableop=savev2_batch_normalization_31_moving_mean_read_readvariableopAsavev2_batch_normalization_31_moving_variance_read_readvariableop=savev2_batch_normalization_32_moving_mean_read_readvariableopAsavev2_batch_normalization_32_moving_variance_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop>savev2_adam_batch_normalization_31_gamma_m_read_readvariableop=savev2_adam_batch_normalization_31_beta_m_read_readvariableop2savev2_adam_conv2d_60_kernel_m_read_readvariableop0savev2_adam_conv2d_60_bias_m_read_readvariableop2savev2_adam_conv2d_61_kernel_m_read_readvariableop0savev2_adam_conv2d_61_bias_m_read_readvariableop>savev2_adam_batch_normalization_32_gamma_m_read_readvariableop=savev2_adam_batch_normalization_32_beta_m_read_readvariableop2savev2_adam_conv2d_62_kernel_m_read_readvariableop0savev2_adam_conv2d_62_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_57_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_57_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_58_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_58_bias_m_read_readvariableop>savev2_adam_batch_normalization_33_gamma_m_read_readvariableop=savev2_adam_batch_normalization_33_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_59_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_59_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_60_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_60_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_61_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_61_bias_m_read_readvariableop>savev2_adam_batch_normalization_31_gamma_v_read_readvariableop=savev2_adam_batch_normalization_31_beta_v_read_readvariableop2savev2_adam_conv2d_60_kernel_v_read_readvariableop0savev2_adam_conv2d_60_bias_v_read_readvariableop2savev2_adam_conv2d_61_kernel_v_read_readvariableop0savev2_adam_conv2d_61_bias_v_read_readvariableop>savev2_adam_batch_normalization_32_gamma_v_read_readvariableop=savev2_adam_batch_normalization_32_beta_v_read_readvariableop2savev2_adam_conv2d_62_kernel_v_read_readvariableop0savev2_adam_conv2d_62_bias_v_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_57_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_57_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_58_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_58_bias_v_read_readvariableop>savev2_adam_batch_normalization_33_gamma_v_read_readvariableop=savev2_adam_batch_normalization_33_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_59_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_59_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_60_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_60_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_61_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_61_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *d
dtypesZ
X2V	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : ::::::::::::::::::::::::::::::: : ::::::::::::::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
:: "

_output_shapes
:: #

_output_shapes
::$

_output_shapes
: :%

_output_shapes
: : &

_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
:: 6

_output_shapes
:: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
:: D

_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
:: M

_output_shapes
:: N

_output_shapes
:: O

_output_shapes
::,P(
&
_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
:: S

_output_shapes
::,T(
&
_output_shapes
:: U

_output_shapes
::V

_output_shapes
: 
?&
?
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5678454

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input6
serving_default_input:0?????????@@C
decoder8
StatefulPartitionedCall:0?????????@@tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
?

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
regularization_losses
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
?
"iter

#beta_1

$beta_2
	%decay
&learning_rate'm?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v?"
	optimizer
 "
trackable_list_wrapper
?
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211
312
413
514
615
716
817
918
:19
;20
<21
=22
>23"
trackable_list_wrapper
?
'0
(1
?2
@3
)4
*5
+6
,7
-8
.9
A10
B11
/12
013
114
215
316
417
518
619
720
821
C22
D23
924
:25
;26
<27
=28
>29"
trackable_list_wrapper
?
regularization_losses

Elayers
Flayer_metrics
trainable_variables
	variables
Gnon_trainable_variables
Hlayer_regularization_losses
Imetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_tf_keras_input_layer
?
Jaxis
	'gamma
(beta
?moving_mean
@moving_variance
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

)kernel
*bias
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

+kernel
,bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Waxis
	-gamma
.beta
Amoving_mean
Bmoving_variance
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

/kernel
0bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
dregularization_losses
etrainable_variables
f	variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
v
'0
(1
)2
*3
+4
,5
-6
.7
/8
09
110
211"
trackable_list_wrapper
?
'0
(1
?2
@3
)4
*5
+6
,7
-8
.9
A10
B11
/12
013
114
215"
trackable_list_wrapper
?
regularization_losses

hlayers
ilayer_metrics
trainable_variables
	variables
jnon_trainable_variables
klayer_regularization_losses
lmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
?
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

3kernel
4bias
qregularization_losses
rtrainable_variables
s	variables
t	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

5kernel
6bias
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
yaxis
	7gamma
8beta
Cmoving_mean
Dmoving_variance
zregularization_losses
{trainable_variables
|	variables
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

9kernel
:bias
~regularization_losses
trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

;kernel
<bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

=kernel
>bias
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
v
30
41
52
63
74
85
96
:7
;8
<9
=10
>11"
trackable_list_wrapper
?
30
41
52
63
74
85
C6
D7
98
:9
;10
<11
=12
>13"
trackable_list_wrapper
?
regularization_losses
?layers
?layer_metrics
trainable_variables
 	variables
?non_trainable_variables
 ?layer_regularization_losses
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2batch_normalization_31/gamma
):'2batch_normalization_31/beta
*:(2conv2d_60/kernel
:2conv2d_60/bias
*:(2conv2d_61/kernel
:2conv2d_61/bias
*:(2batch_normalization_32/gamma
):'2batch_normalization_32/beta
*:(2conv2d_62/kernel
:2conv2d_62/bias
*:(2conv2d_63/kernel
:2conv2d_63/bias
4:22conv2d_transpose_57/kernel
&:$2conv2d_transpose_57/bias
4:22conv2d_transpose_58/kernel
&:$2conv2d_transpose_58/bias
*:(2batch_normalization_33/gamma
):'2batch_normalization_33/beta
4:22conv2d_transpose_59/kernel
&:$2conv2d_transpose_59/bias
4:22conv2d_transpose_60/kernel
&:$2conv2d_transpose_60/bias
4:22conv2d_transpose_61/kernel
&:$2conv2d_transpose_61/bias
2:0 (2"batch_normalization_31/moving_mean
6:4 (2&batch_normalization_31/moving_variance
2:0 (2"batch_normalization_32/moving_mean
6:4 (2&batch_normalization_32/moving_variance
2:0 (2"batch_normalization_33/moving_mean
6:4 (2&batch_normalization_33/moving_variance
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
J
?0
@1
A2
B3
C4
D5"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
<
'0
(1
?2
@3"
trackable_list_wrapper
?
Kregularization_losses
?layers
Ltrainable_variables
M	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
Oregularization_losses
?layers
Ptrainable_variables
Q	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
Sregularization_losses
?layers
Ttrainable_variables
U	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
<
-0
.1
A2
B3"
trackable_list_wrapper
?
Xregularization_losses
?layers
Ytrainable_variables
Z	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
\regularization_losses
?layers
]trainable_variables
^	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
`regularization_losses
?layers
atrainable_variables
b	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
dregularization_losses
?layers
etrainable_variables
f	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mregularization_losses
?layers
ntrainable_variables
o	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
qregularization_losses
?layers
rtrainable_variables
s	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
uregularization_losses
?layers
vtrainable_variables
w	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
<
70
81
C2
D3"
trackable_list_wrapper
?
zregularization_losses
?layers
{trainable_variables
|	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
~regularization_losses
?layers
trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
?regularization_losses
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?regularization_losses
?layers
?trainable_variables
?	variables
 ?layer_regularization_losses
?non_trainable_variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-2#Adam/batch_normalization_31/gamma/m
.:,2"Adam/batch_normalization_31/beta/m
/:-2Adam/conv2d_60/kernel/m
!:2Adam/conv2d_60/bias/m
/:-2Adam/conv2d_61/kernel/m
!:2Adam/conv2d_61/bias/m
/:-2#Adam/batch_normalization_32/gamma/m
.:,2"Adam/batch_normalization_32/beta/m
/:-2Adam/conv2d_62/kernel/m
!:2Adam/conv2d_62/bias/m
/:-2Adam/conv2d_63/kernel/m
!:2Adam/conv2d_63/bias/m
9:72!Adam/conv2d_transpose_57/kernel/m
+:)2Adam/conv2d_transpose_57/bias/m
9:72!Adam/conv2d_transpose_58/kernel/m
+:)2Adam/conv2d_transpose_58/bias/m
/:-2#Adam/batch_normalization_33/gamma/m
.:,2"Adam/batch_normalization_33/beta/m
9:72!Adam/conv2d_transpose_59/kernel/m
+:)2Adam/conv2d_transpose_59/bias/m
9:72!Adam/conv2d_transpose_60/kernel/m
+:)2Adam/conv2d_transpose_60/bias/m
9:72!Adam/conv2d_transpose_61/kernel/m
+:)2Adam/conv2d_transpose_61/bias/m
/:-2#Adam/batch_normalization_31/gamma/v
.:,2"Adam/batch_normalization_31/beta/v
/:-2Adam/conv2d_60/kernel/v
!:2Adam/conv2d_60/bias/v
/:-2Adam/conv2d_61/kernel/v
!:2Adam/conv2d_61/bias/v
/:-2#Adam/batch_normalization_32/gamma/v
.:,2"Adam/batch_normalization_32/beta/v
/:-2Adam/conv2d_62/kernel/v
!:2Adam/conv2d_62/bias/v
/:-2Adam/conv2d_63/kernel/v
!:2Adam/conv2d_63/bias/v
9:72!Adam/conv2d_transpose_57/kernel/v
+:)2Adam/conv2d_transpose_57/bias/v
9:72!Adam/conv2d_transpose_58/kernel/v
+:)2Adam/conv2d_transpose_58/bias/v
/:-2#Adam/batch_normalization_33/gamma/v
.:,2"Adam/batch_normalization_33/beta/v
9:72!Adam/conv2d_transpose_59/kernel/v
+:)2Adam/conv2d_transpose_59/bias/v
9:72!Adam/conv2d_transpose_60/kernel/v
+:)2Adam/conv2d_transpose_60/bias/v
9:72!Adam/conv2d_transpose_61/kernel/v
+:)2Adam/conv2d_transpose_61/bias/v
?2?
-__inference_autoencoder_layer_call_fn_5679650
-__inference_autoencoder_layer_call_fn_5680181
-__inference_autoencoder_layer_call_fn_5680246
-__inference_autoencoder_layer_call_fn_5679911?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_5677617input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680431
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680616
H__inference_autoencoder_layer_call_and_return_conditional_losses_5679977
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680043?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_encoder_layer_call_fn_5678042
)__inference_encoder_layer_call_fn_5680653
)__inference_encoder_layer_call_fn_5680690
)__inference_encoder_layer_call_fn_5678330?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_encoder_layer_call_and_return_conditional_losses_5680752
D__inference_encoder_layer_call_and_return_conditional_losses_5680814
D__inference_encoder_layer_call_and_return_conditional_losses_5678373
D__inference_encoder_layer_call_and_return_conditional_losses_5678416?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_decoder_layer_call_fn_5679206
)__inference_decoder_layer_call_fn_5680847
)__inference_decoder_layer_call_fn_5680880
)__inference_decoder_layer_call_fn_5679439?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_decoder_layer_call_and_return_conditional_losses_5681007
D__inference_decoder_layer_call_and_return_conditional_losses_5681134
D__inference_decoder_layer_call_and_return_conditional_losses_5679478
D__inference_decoder_layer_call_and_return_conditional_losses_5679517?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_signature_wrapper_5680116input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_31_layer_call_fn_5681147
8__inference_batch_normalization_31_layer_call_fn_5681160
8__inference_batch_normalization_31_layer_call_fn_5681173
8__inference_batch_normalization_31_layer_call_fn_5681186?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681204
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681222
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681240
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681258?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_conv2d_60_layer_call_fn_5681267?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_5681278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_61_layer_call_fn_5681287?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_5681298?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_32_layer_call_fn_5681311
8__inference_batch_normalization_32_layer_call_fn_5681324
8__inference_batch_normalization_32_layer_call_fn_5681337
8__inference_batch_normalization_32_layer_call_fn_5681350?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681368
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681386
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681404
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681422?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_conv2d_62_layer_call_fn_5681431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_5681442?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_63_layer_call_fn_5681451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_5681462?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_8_layer_call_fn_5681467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_8_layer_call_and_return_conditional_losses_5681473?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_reshape_8_layer_call_fn_5681478?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_8_layer_call_and_return_conditional_losses_5681492?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv2d_transpose_57_layer_call_fn_5681501
5__inference_conv2d_transpose_57_layer_call_fn_5681510?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5681544
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5681568?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv2d_transpose_58_layer_call_fn_5681577
5__inference_conv2d_transpose_58_layer_call_fn_5681586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5681620
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5681644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_batch_normalization_33_layer_call_fn_5681657
8__inference_batch_normalization_33_layer_call_fn_5681670
8__inference_batch_normalization_33_layer_call_fn_5681683
8__inference_batch_normalization_33_layer_call_fn_5681696?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681714
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681732
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681750
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681768?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_conv2d_transpose_59_layer_call_fn_5681777
5__inference_conv2d_transpose_59_layer_call_fn_5681786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5681820
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5681844?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv2d_transpose_60_layer_call_fn_5681853
5__inference_conv2d_transpose_60_layer_call_fn_5681862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5681896
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5681920?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv2d_transpose_61_layer_call_fn_5681929
5__inference_conv2d_transpose_61_layer_call_fn_5681938?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5681971
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5681994?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_5677617?'(?@)*+,-.AB/012345678CD9:;<=>6?3
,?)
'?$
input?????????@@
? "9?6
4
decoder)?&
decoder?????????@@?
H__inference_autoencoder_layer_call_and_return_conditional_losses_5679977?'(?@)*+,-.AB/012345678CD9:;<=>>?;
4?1
'?$
input?????????@@
p 

 
? "-?*
#? 
0?????????@@
? ?
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680043?'(?@)*+,-.AB/012345678CD9:;<=>>?;
4?1
'?$
input?????????@@
p

 
? "-?*
#? 
0?????????@@
? ?
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680431?'(?@)*+,-.AB/012345678CD9:;<=>??<
5?2
(?%
inputs?????????@@
p 

 
? "-?*
#? 
0?????????@@
? ?
H__inference_autoencoder_layer_call_and_return_conditional_losses_5680616?'(?@)*+,-.AB/012345678CD9:;<=>??<
5?2
(?%
inputs?????????@@
p

 
? "-?*
#? 
0?????????@@
? ?
-__inference_autoencoder_layer_call_fn_5679650?'(?@)*+,-.AB/012345678CD9:;<=>>?;
4?1
'?$
input?????????@@
p 

 
? " ??????????@@?
-__inference_autoencoder_layer_call_fn_5679911?'(?@)*+,-.AB/012345678CD9:;<=>>?;
4?1
'?$
input?????????@@
p

 
? " ??????????@@?
-__inference_autoencoder_layer_call_fn_5680181?'(?@)*+,-.AB/012345678CD9:;<=>??<
5?2
(?%
inputs?????????@@
p 

 
? " ??????????@@?
-__inference_autoencoder_layer_call_fn_5680246?'(?@)*+,-.AB/012345678CD9:;<=>??<
5?2
(?%
inputs?????????@@
p

 
? " ??????????@@?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681204?'(?@M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681222?'(?@M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681240r'(?@;?8
1?.
(?%
inputs?????????@@
p 
? "-?*
#? 
0?????????@@
? ?
S__inference_batch_normalization_31_layer_call_and_return_conditional_losses_5681258r'(?@;?8
1?.
(?%
inputs?????????@@
p
? "-?*
#? 
0?????????@@
? ?
8__inference_batch_normalization_31_layer_call_fn_5681147?'(?@M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_31_layer_call_fn_5681160?'(?@M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_31_layer_call_fn_5681173e'(?@;?8
1?.
(?%
inputs?????????@@
p 
? " ??????????@@?
8__inference_batch_normalization_31_layer_call_fn_5681186e'(?@;?8
1?.
(?%
inputs?????????@@
p
? " ??????????@@?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681368?-.ABM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681386?-.ABM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681404r-.AB;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_32_layer_call_and_return_conditional_losses_5681422r-.AB;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
8__inference_batch_normalization_32_layer_call_fn_5681311?-.ABM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_32_layer_call_fn_5681324?-.ABM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_32_layer_call_fn_5681337e-.AB;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
8__inference_batch_normalization_32_layer_call_fn_5681350e-.AB;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681714?78CDM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681732?78CDM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681750r78CD;?8
1?.
(?%
inputs?????????  
p 
? "-?*
#? 
0?????????  
? ?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_5681768r78CD;?8
1?.
(?%
inputs?????????  
p
? "-?*
#? 
0?????????  
? ?
8__inference_batch_normalization_33_layer_call_fn_5681657?78CDM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
8__inference_batch_normalization_33_layer_call_fn_5681670?78CDM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
8__inference_batch_normalization_33_layer_call_fn_5681683e78CD;?8
1?.
(?%
inputs?????????  
p 
? " ??????????  ?
8__inference_batch_normalization_33_layer_call_fn_5681696e78CD;?8
1?.
(?%
inputs?????????  
p
? " ??????????  ?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_5681278l)*7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
+__inference_conv2d_60_layer_call_fn_5681267_)*7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_5681298l+,7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_61_layer_call_fn_5681287_+,7?4
-?*
(?%
inputs?????????@@
? " ??????????  ?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_5681442l/07?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
+__inference_conv2d_62_layer_call_fn_5681431_/07?4
-?*
(?%
inputs?????????  
? " ??????????  ?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_5681462l127?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????
? ?
+__inference_conv2d_63_layer_call_fn_5681451_127?4
-?*
(?%
inputs?????????  
? " ???????????
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5681544?34I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_57_layer_call_and_return_conditional_losses_5681568l347?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
5__inference_conv2d_transpose_57_layer_call_fn_5681501?34I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_57_layer_call_fn_5681510_347?4
-?*
(?%
inputs?????????
? " ???????????
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5681620?56I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_58_layer_call_and_return_conditional_losses_5681644l567?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????  
? ?
5__inference_conv2d_transpose_58_layer_call_fn_5681577?56I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_58_layer_call_fn_5681586_567?4
-?*
(?%
inputs?????????
? " ??????????  ?
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5681820?9:I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_59_layer_call_and_return_conditional_losses_5681844l9:7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????  
? ?
5__inference_conv2d_transpose_59_layer_call_fn_5681777?9:I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_59_layer_call_fn_5681786_9:7?4
-?*
(?%
inputs?????????  
? " ??????????  ?
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5681896?;<I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_60_layer_call_and_return_conditional_losses_5681920l;<7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????@@
? ?
5__inference_conv2d_transpose_60_layer_call_fn_5681853?;<I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_60_layer_call_fn_5681862_;<7?4
-?*
(?%
inputs?????????  
? " ??????????@@?
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5681971?=>I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
P__inference_conv2d_transpose_61_layer_call_and_return_conditional_losses_5681994l=>7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
5__inference_conv2d_transpose_61_layer_call_fn_5681929?=>I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
5__inference_conv2d_transpose_61_layer_call_fn_5681938_=>7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
D__inference_decoder_layer_call_and_return_conditional_losses_5679478?345678CD9:;<=>??<
5?2
(?%
encoded audio??????????
p 

 
? "-?*
#? 
0?????????@@
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_5679517?345678CD9:;<=>??<
5?2
(?%
encoded audio??????????
p

 
? "-?*
#? 
0?????????@@
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_5681007y345678CD9:;<=>8?5
.?+
!?
inputs??????????
p 

 
? "-?*
#? 
0?????????@@
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_5681134y345678CD9:;<=>8?5
.?+
!?
inputs??????????
p

 
? "-?*
#? 
0?????????@@
? ?
)__inference_decoder_layer_call_fn_5679206s345678CD9:;<=>??<
5?2
(?%
encoded audio??????????
p 

 
? " ??????????@@?
)__inference_decoder_layer_call_fn_5679439s345678CD9:;<=>??<
5?2
(?%
encoded audio??????????
p

 
? " ??????????@@?
)__inference_decoder_layer_call_fn_5680847l345678CD9:;<=>8?5
.?+
!?
inputs??????????
p 

 
? " ??????????@@?
)__inference_decoder_layer_call_fn_5680880l345678CD9:;<=>8?5
.?+
!?
inputs??????????
p

 
? " ??????????@@?
D__inference_encoder_layer_call_and_return_conditional_losses_5678373?'(?@)*+,-.AB/012G?D
=?:
0?-
original audio?????????@@
p 

 
? "&?#
?
0??????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_5678416?'(?@)*+,-.AB/012G?D
=?:
0?-
original audio?????????@@
p

 
? "&?#
?
0??????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_5680752{'(?@)*+,-.AB/012??<
5?2
(?%
inputs?????????@@
p 

 
? "&?#
?
0??????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_5680814{'(?@)*+,-.AB/012??<
5?2
(?%
inputs?????????@@
p

 
? "&?#
?
0??????????
? ?
)__inference_encoder_layer_call_fn_5678042v'(?@)*+,-.AB/012G?D
=?:
0?-
original audio?????????@@
p 

 
? "????????????
)__inference_encoder_layer_call_fn_5678330v'(?@)*+,-.AB/012G?D
=?:
0?-
original audio?????????@@
p

 
? "????????????
)__inference_encoder_layer_call_fn_5680653n'(?@)*+,-.AB/012??<
5?2
(?%
inputs?????????@@
p 

 
? "????????????
)__inference_encoder_layer_call_fn_5680690n'(?@)*+,-.AB/012??<
5?2
(?%
inputs?????????@@
p

 
? "????????????
F__inference_flatten_8_layer_call_and_return_conditional_losses_5681473a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
+__inference_flatten_8_layer_call_fn_5681467T7?4
-?*
(?%
inputs?????????
? "????????????
F__inference_reshape_8_layer_call_and_return_conditional_losses_5681492a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
+__inference_reshape_8_layer_call_fn_5681478T0?-
&?#
!?
inputs??????????
? " ???????????
%__inference_signature_wrapper_5680116?'(?@)*+,-.AB/012345678CD9:;<=>??<
? 
5?2
0
input'?$
input?????????@@"9?6
4
decoder)?&
decoder?????????@@