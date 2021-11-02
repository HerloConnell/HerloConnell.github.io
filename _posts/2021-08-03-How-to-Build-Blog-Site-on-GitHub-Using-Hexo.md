---
layout: post
title: How to Build Blog Site on GitHub Using Hexo
date: 2021-08-03 17:51
comments: true
external-url:
categories: Others
---

## 安装：Git、node.js、 hexo

1. 安装Homebrew

2. 安装node.js

   ```
   brew install node
   ```

3. 安装git

   一种方法是安装Xcode, 另一种方法是

   ```
   brew install git
   ```

4. 使用 npm 安装 hexo

   ```
   npm install -g hexo-cli
   ```

## 初始化博客

1. 在某位置新建blogs文件夹，如`～/Documents/blogs`，并`cd`入该文件夹

2. 博客初始化，这一步会在`blogs`中生成一些配置文件

   ```
   hexo init
   ```

3. 进行本地预览

   ```
   hexo s
   ```

4. 在`http://localhost:4000`进行预览，看到HEXO的`hello world`界面即为成功

## 添加SSH Key到github

`Enter file in which to save the key`时直接按回车，即在默认位置生成ssh文件，即`/Users/xxx/.ssh/id_rsa.pub`

`Enter passphrase`时直接回车即不需要密码，如设置密码，该密码为push时需要的密码，与github账户密码等无关

```
% git config --global user.name "github账户名，大小写敏感"                   
% git config --global user.email "github账户邮箱"
% ssh-keygen -t rsa -C "github账户邮箱"
Generating public/private rsa key pair.
Enter file in which to save the key (/Users/xxx/.ssh/id_rsa): 
Created directory '/Users/lihan/.ssh'.
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /Users/xxx/.ssh/id_rsa.
Your public key has been saved in /Users/xxx/.ssh/id_rsa.pub.

```

生成文件之后，复制`id_rsa.pub`中的内容，这就是需要的key；登陆github -> setting -> SSH and GPG keys -> New SSH key，将key复制粘贴，Title可以取名为设备名

## 本地博客关联到Github主页

1. 登录Github并且创建一个名字为 `username.github.io` 的仓库，如`HerloConnell.github.io`，选择`Public`

2. 修改`blogs/_config.yml`的`deploy`，注意这里使用的是分支master

   **在8.13号以后git需要使用令牌，repo设置稍有改动，具体见错误2**

   ```yaml
   deploy:
     type: git
     repo: https://github.com/HerloConnell/HerloConnell.github.io.git
     branch: master
   ```

3. 将博客push到GitHub

   ```
   hexo g
   hexo d
   ```

4. 在该仓库的`settings`中选择`GitHub Pages`,点击`heck it out here!`，显示

   ```
   Your site is published at https://herloconnell.github.io/
   ```

   在`Source`处选择`Branch:master`

5. 再次执行下面的代码，此时应该可以在`https://herloconnell.github.io/`看到Hexo界面

   ```
   hexo g
   hexo d
   ```

## hexo常用命令

```
hexo n "博客名称"  => hexo new "博客名称"   #这两个都是创建新文章，前者是简写模式
hexo p  => hexo publish
hexo g  => hexo generate  #生成
hexo s  => hexo server  #启动服务预览
hexo d  => hexo deploy  #部署  

hexo server   #Hexo 会监视文件变动并自动更新，无须重启服务器。
hexo server -s   #静态模式
hexo server -p 5000   #更改端口
hexo server -i 192.168.1.1   #自定义IP
hexo clean   #清除缓存，网页正常情况下可以忽略此条命令
hexo g   #生成静态网页
hexo d   #开始部署
```

## Fluid主题

[hexo主题网站](https://hexo.io/themes/)，[Fluid](https://github.com/fluid-dev/hexo-theme-fluid)

1. 在`/blogs/themes`中

   ```bash
   git clone https://github.com/fluid-dev/hexo-theme-fluid.git
   ```

2. 将`hexo-theme-fluid`文件夹更名为`fluid`，并修改hexo的`_config.yml`

   ```yaml
   theme: fluid
   ```

3. 参考`Fluid`主题配置进行配置，如配置`Latex`支持等（注：`Latex`需要在文章 [Front-matter](https://hexo.io/zh-cn/docs/front-matter)里指定 `math: true` 才会在文章页启动公式转换，以便在页面不包含公式时提高加载速度）

## Twikoo 评论系统

[Twikoo官网 ](https://twikoo.js.org/quick-start.html#云函数部署)，[参考于该篇博客](https://www.anzifan.com/post/icarus_to_candy_2/)

1. 在腾讯云申请云环境

   > [注册云开发CloudBase](https://curl.qcloud.com/KnnJtUom)
   >
   > 进入云开发控制台，新建环境，请按需配置环境：环境名称自由填写，选择计费方式`包年包月`，套餐版本`免费版`，超出免费额度不会收费
   >
   > 进入[环境-登录授权](https://console.cloud.tencent.com/tcb/env/login)，启用“匿名登录”；
   >
   > 进入[环境-安全配置](https://console.cloud.tencent.com/tcb/env/safety)，将网站域名(如`herloconnell.github.io`)添加到“WEB安全域名”

2. 配置云函数

   > 进入[环境-云函数](https://console.cloud.tencent.com/tcb/scf/index)，点击“新建云函数”
   >
   > 函数名称请填写：`twikoo`，创建方式请选择：`空白函数`，运行环境请选择：`Nodejs 10.15`，函数内存请选择：`128MB`，点击“下一步”
   >
   > 打开 [index.js](https://imaegoo.coding.net/public/twikoo/twikoo/git/files/dev/src/function/twikoo/index.js)，全选代码、复制、粘贴到“函数代码”输入框中，点击“确定”
   >
   > 创建完成后，点击“twikoo"进入云函数详情页，进入“函数代码”标签，点击“文件 - 新建文件”，输入 `package.json`，回车
   >
   > 打开 [package.json](https://imaegoo.coding.net/public/twikoo/twikoo/git/files/dev/src/function/twikoo/package.json)，全选代码、复制、粘贴到代码框中，点击“保存并安装依赖”

3. 修改config.yml

   ```yaml
   comments:
   enable: true
   # 指定的插件，需要同时设置对应插件的必要参数
   # The specified plugin needs to set the necessary parameters at the same time
   # Options: utterances | disqus | gitalk | valine | waline | changyan | livere | remark42 | twikoo | cusdis
   type: twikoo
   ....
   twikoo:
     envId: #云环境ID
     region: # ap-guangzhou 或 ap-shanghai，根据云环境申请地而定
     path: # window.location.pathname
   ```

4. 配置、管理你的 Twikoo 评论系统

   到这一步`hexo clean`，`hexo g`，`hexo d`之后应该可以看到评论系统。

   >进入[环境-登录授权](https://console.cloud.tencent.com/tcb/env/login)，点击“自定义登录”右边的“私钥下载”，下载私钥文件
   >
   >用文本编辑器打开私钥文件，复制全部内容
   >
   >你只需要来到最下面的评论区，点击小齿轮 即可配置更多设置，当然博主是需要登录的，初次进入需要粘贴私钥文件内容，并设置管理员密码。

## 错误

**错误一：443**


```bash
fatal unable to access https://github.com LibreSSL SSL_connect SSL_ERROR_SYSCALL in connection to github.com 443
```

科学上网的换个节点或者多提交几次即可

**错误二：403**

```bash
remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.
remote: Please see https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/ for more information.
fatal: unable to access 'https://github.com/HerloConnell/HerloConnell.github.io.git/': The requested URL returned error: 403
```

2021年8月13后 GitHub 不能再用密码pull/push 需要使用令牌。

- 进入 Github --> Settings --> Developer settings --> Personal access tokens获得一个token，这里作用域我点的全选。

- 修改`config`

  ```yaml
  deploy:
    type: git
    repo: https://<Token>@github.com/HerloConnell/HerloConnell.github.io.git
    branch: master
  ```

  

   



