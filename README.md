# Code samples for "Neural Networks and Deep Learning"


此代码以mnielsen大神的教程示例代码为基础，做了如下扩展：
- 对dnn的层级做了抽象，包括layer层，output层；
- 对activation函数和cost函数做了抽象，可以很方便的自定义和替换；
- 实现了tanh的激励函数，实现了softmax+极大似然的损失函数；
- 实现了sgd中对batch做矩阵操作；

folder：src/lanpay_network

