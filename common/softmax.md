可以从最普通的 softmax 定义一步一步推导到 LSE 形式。

Softmax 对一组 logits \(x_1,\dots,x_n\) 的定义是：

\[
\mathrm{softmax}(x_i)
=
\frac{e^{x_i}}
{\sum_{j=1}^{n}e^{x_j}}
\]

为了简化，先记：

\[
Z=\sum_{j=1}^{n} e^{x_j}
\]

那么：

\[
\mathrm{softmax}(x_i)=\frac{e^{x_i}}{Z}
\]

现在定义 LSE，也就是 LogSumExp：

\[
\boxed{
LSE(x)=\log\left(\sum_{j=1}^{n}e^{x_j}\right)
}
\]

因为：

\[
Z=\sum_j e^{x_j}
\]

所以：

\[
LSE=\log Z
\]

对两边取指数：

\[
e^{LSE}=e^{\log Z}=Z
\]

因此：

\[
\boxed{
Z=e^{LSE}
}
\]

把它代回 softmax：

\[
\mathrm{softmax}(x_i)
=
\frac{e^{x_i}}{e^{LSE}}
\]

利用指数运算法则：

\[
\frac{e^a}{e^b}=e^{a-b}
\]

于是得到：

\[
\boxed{
\mathrm{softmax}(x_i)=e^{x_i-LSE}
}
\]

这就是 softmax 使用 LSE 计算的完整数学来源。

---

举个具体例子。

假设：

\[
x=[2,1,0]
\]

普通 softmax 是：

\[
p_1=
\frac{e^2}{e^2+e^1+e^0}
\]

\[
p_2=
\frac{e^1}{e^2+e^1+e^0}
\]

\[
p_3=
\frac{e^0}{e^2+e^1+e^0}
\]

先算分母：

\[
Z=e^2+e^1+e^0
\]

\[
=7.389+2.718+1
=11.107
\]

所以：

\[
LSE=\log Z=\log 11.107\approx2.4076
\]

现在改成 LSE 形式：

\[
p_1=e^{2-2.4076}
\]

\[
p_2=e^{1-2.4076}
\]

\[
p_3=e^{0-2.4076}
\]

得到：

\[
p_1\approx0.6652
\]

\[
p_2\approx0.2447
\]

\[
p_3\approx0.0900
\]

和直接 softmax 完全一样。

---

更值得理解的是，为什么实际实现里喜欢 LSE。

因为直接计算：

\[
e^{x_i}
\]

可能数值溢出。

例如：

\[
x=[1000,999,998]
\]

理论 softmax：

\[
\frac{e^{1000}}
{e^{1000}+e^{999}+e^{998}}
\]

但在 float32 里：

\[
e^{1000}
\]

早就溢出了。

所以先做一个标准变换：

\[
m=\max_i x_i
\]

这里：

\[
m=1000
\]

把所有 logits 同时减去 \(m\)。

注意 softmax 有一个非常重要的性质：

\[
\boxed{
softmax(x_i)=softmax(x_i-c)
}
\]

因为：

\[
\frac{e^{x_i-c}}
{\sum_j e^{x_j-c}}
\]

分子：

\[
e^{x_i-c}=e^{x_i}e^{-c}
\]

分母：

\[
\sum_j e^{x_j-c}
=
e^{-c}\sum_j e^{x_j}
\]

所以：

\[
\frac{e^{x_i}e^{-c}}
{e^{-c}\sum_j e^{x_j}}
=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
\]

公共因子 \(e^{-c}\) 被约掉了。

因此我们可以安全地减最大值：

\[
x-m=[0,-1,-2]
\]

这时候只需要计算：

\[
e^0,\quad e^{-1},\quad e^{-2}
\]

都非常安全。

---

现在看 LSE 怎么跟这个“减最大值”结合起来。

原始定义：

\[
LSE
=
\log\sum_j e^{x_j}
\]

把：

\[
x_j=(x_j-m)+m
\]

代进去：

\[
LSE
=
\log
\sum_j
e^{(x_j-m)+m}
\]

利用：

\[
e^{a+b}=e^a e^b
\]

得到：

\[
LSE
=
\log
\sum_j
e^{x_j-m}e^m
\]

因为 \(e^m\) 和 \(j\) 无关，可以提出来：

\[
LSE
=
\log
\left(
e^m
\sum_j e^{x_j-m}
\right)
\]

利用：

\[
\log(ab)=\log a+\log b
\]

得到：

\[
LSE
=
\log e^m
+
\log\sum_j e^{x_j-m}
\]

而：

\[
\log e^m=m
\]

所以：

\[
\boxed{
LSE
=
m+
\log\sum_j e^{x_j-m}
}
\]

这就是实际实现中使用的稳定版 LogSumExp。

对于：

\[
[1000,999,998]
\]

有：

\[
m=1000
\]

所以：

\[
LSE
=
1000+
\log
\left(
e^0+e^{-1}+e^{-2}
\right)
\]

数值：

\[
e^0+e^{-1}+e^{-2}
=
1+0.3679+0.1353
=
1.5032
\]

因此：

\[
LSE
=
1000+\log(1.5032)
\]

\[
\approx1000.4076
\]

最后 softmax：

\[
p_i=e^{x_i-LSE}
\]

于是：

\[
p_1=e^{1000-1000.4076}
=e^{-0.4076}
\approx0.6652
\]

\[
p_2=e^{-1.4076}
\approx0.2447
\]

\[
p_3=e^{-2.4076}
\approx0.0900
\]

完全避免了：

\[
e^{1000}
\]

这样的危险计算。

---

你还可以从 log-softmax 的角度理解。

因为：

\[
softmax_i
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

两边取 log：

\[
\log softmax_i
=
\log e^{x_i}
-
\log\sum_j e^{x_j}
\]

所以：

\[
\boxed{
\log softmax_i
=
x_i-LSE
}
\]

于是：

\[
\boxed{
softmax_i
=
e^{x_i-LSE}
}
\]

所以实际上：

\[
\boxed{
logsoftmax(x)=x-LSE(x)
}
\]

这是一个特别重要的关系。

---

放到 Attention 里面就更直观了。

对于一个 query：

\[
q
\]

对所有 key 的 score：

\[
s_j=\frac{qk_j^T}{\sqrt d}
\]

普通 attention probability：

\[
p_j
=
\frac{e^{s_j}}
{\sum_k e^{s_k}}
\]

定义：

\[
LSE_q
=
\log\sum_k e^{s_k}
\]

那么：

\[
\boxed{
p_j=e^{s_j-LSE_q}
}
\]

所以每一个 query、每一个 head，都对应一个 LSE：

\[
LSE_{q,h}
\]

它就是这一行 attention logits 的 softmax normalizer 的 log。

例如 attention matrix 某个 head：

\[
S=
\begin{bmatrix}
2&1&0\\
5&2&-1
\end{bmatrix}
\]

第一行对应 query 1：

\[
LSE_1=\log(e^2+e^1+e^0)
\]

第二行对应 query 2：

\[
LSE_2=\log(e^5+e^2+e^{-1})
\]

然后：

\[
P_{1,j}=e^{S_{1,j}-LSE_1}
\]

\[
P_{2,j}=e^{S_{2,j}-LSE_2}
\]

所以 LSE 是**逐 query 行计算**的，而不是整个 attention matrix 只有一个值。

---

最后可以把整个关系压缩成：

\[
\boxed{
Z=\sum_j e^{x_j}
}
\]

\[
\boxed{
LSE=\log Z
}
\]

因此：

\[
\boxed{
Z=e^{LSE}
}
\]

所以：

\[
\frac{e^{x_i}}{Z}
=
\frac{e^{x_i}}{e^{LSE}}
=
\boxed{e^{x_i-LSE}}
\]

也就是说，**LSE 并没有改变 softmax，它只是把 softmax 的分母从普通数值域搬到了 log 域里表示。**

这也是为什么后面 attention sink 只需要把

\[
LSE=\log\sum_j e^{s_j}
\]

替换成

\[
LSE_{\text{sink}}
=
\log\left(
\sum_j e^{s_j}+e^{sink}
\right)
\]

就能自然把 sink 放进 softmax 分母。