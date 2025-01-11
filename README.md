# LibFewShot
LibFewShot HomeWork


# 课程大作业-小样本学习 
```
概述：在https://github.com/RL-VIG/LibFewShot框架上实现⼀种尚未在该库中集成的⼩样本算法，该算法
需要为已发表的顶会论⽂，复现精度误差为~2%以内。
```

# 作业要求
- 在【腾讯文档】Lib论文选择
https://docs.qq.com/sheet/DWFV5SnBEV2licmdj 

- 最后选择此篇论文 https://github.com/han-jia/unicorn-maml ， 名字为 "Task-aware Part Mining Network for Few-Shot Learning"

- 选择一篇小样本论文进 行复现，特殊情况自行选择近几年顶会经典的小样本学习算法（需与助教商量） 需要在
https://github.com/RL-VIG/LibFewShot  (也就是该框架下)
框架基础上完成算法复现，与论文中汇报的精度误差在2%以内，代码规范

- 完成实验报告，报告内陈述：对算法的理解，算法复现的难点以及如何解决，复现结果表格，实验报告
无字数要求，提交时将 **实验报告+训练后模型+增加了所复现代码的代码包**三者打包的压缩包（只包含复
现成功的模型和实验结果，如果压缩包太大请检查是否有太多冗余内容）发送至 nju_ml@163.com，
一个队伍提交一份

- 提交内容命名要求:  秋2024_libfewshot_队伍编号.rar （邮件主题名去掉.rar）


# 环境要求 
- 带有CUDA⽀持的PyTorch环境 
- 安装及环境测试，可以参考 https://pytorch.org/get-started/locally/#linux-installation
- LibFewShot只在Linux+CUDA上测试过，因此不保证在Windows环境下没有使⽤问题

# 数据集 
- 小样本分类⼀般使⽤miniImageNet、tieredImageNet等数据集 
- 下载链接在：
https://github.com/RL-VIG/LibFewShot#datasets， 或者使⽤
https://box.nju.edu.cn/d/7f6c5bd7cfaf4b019c34/
- 下载完之后解压到你喜欢的⽬录就可以

# 框架使用 
- 代码设置、安装、简要介绍请参考
 https://libfewshot-en.readthedocs.io/zh_CN/latest/install.html
- /LibFewShot/reproduce/ 目录下包含已有方法对应的配置参数文件，可以用作测试或者自己方法配置
文件的基础（非常具有指导意义！）

# Q&A 
1. 我没有⽀持CUDA的电脑/服务器可以⽤，怎么办？ 
- 如果你的电脑有NVIDIA的显卡，那⼀般是可以安装CUDA的，请参考 https://developer.nvidia.com/cuda-toolkit。
- 可以使⽤Google的Colab或者Azure的免费服务器
- 如果你的电脑有AMD的显卡并且你的硬件可以在硬软件⽀持中找到，那么你可以尝试安装ROCm
版本的PyTorch， 也可以使⽤。
- 如果你使⽤的是M系列芯⽚的Macbook，PyTorch已经⽀持MPS后端，虽然LibFewShot还没有在
该环境下测试过， 但应该兼容，可能会有点⼩问题需要解决。 
- 如果以上解决⽅案对你⽽⾔都⽐较难，请看最后⼀个Q&A

 2. 我找到了⼀个算法，GitHub已经有他的官⽅实现/第三⽅实现了，我可以借鉴吗？ 
- 可以，这会减轻很多的复现难度，但请注意以下⼏点： 这份实现是不是正确的？包括但不限于： 1）我使⽤这个代码，按照对应的配置⽂件，能不能跑出原⽂声称的 结果？ 
2）仓库的issue列表
⾥，有没有对复现结果的争议？
3）算法实现细节是否和原⽂描述的⼀致？ 

- 这份代码如果是TensorFlow或者MXNet等实现的，那么在参考复现的过程中，需要对⽐与
PyTorch间操作的差 异。 
- 如果有核⼼代码的借鉴，需要在所增加的分类头⽂件的开始，添加原仓库的License或者来源声
明。⿎励在原 实现上进⾏精简和优化，例如PyTorch内置了很多奇怪的函数可以快速地解决某些
复杂运算

3. 我实现了论⽂的算法，但是训练完之后精度很低，这是为什么？ 请从以下⼏个⽅⾯排查原因：
- 检查训练时载⼊的参数，例如学习率等，是否和原⽂/原代码⼀致？ 
- 检查训练和测试时数据增⼴是否和原⽂⼀致？ 
- 原⽂是否使⽤了预训练模型？
- 实现上的问题

4. 我发现了LibFewShot的⼀个BUG！我该怎么反馈？
- 你可以在库的issue⾥直接提出来，并附上对该bug的描述。如果你还不确定这是不是⼀个bug，
请看最后⼀个 Q&A。

5. 我还有其他的问题，怎么办？
- 咨询助教