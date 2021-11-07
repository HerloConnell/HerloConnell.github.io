---
layout: post
title: How to Build Blog Site on GitHub Using Jekyll
date: 2021-11-07 15:39
comments: true
external-url:
categories: Others
---

最近无意中看到Jekyll，我以前的blog是基于Hexo的，上传到github之后，并没有源文件，文件结构我自己都搞不清楚；干脆趁这次机会更换为Jekyll，选择了一个更为简洁的主题，方便自己对文件结构有足够的了解，且github仓库中可以保留原`md`文件。(注：基于Mac OS)

## 下载Ruby

需要安装Ruby, Mac已经有内置的ruby，版本较低，现用brew再下载一个 ruby3.0，并添加到配置文件中

```text
lihan@LideMacBook-Air ~ % brew install ruby
==> Installing dependencies for ruby: libyaml, openssl@1.1 and readline
==> Installing ruby dependency: libyaml
==> Pouring libyaml--0.2.5.arm64_big_sur.bottle.tar.gz
🍺  /opt/homebrew/Cellar/libyaml/0.2.5: 10 files, 369.9KB
==> Installing ruby dependency: openssl@1.1
==> Pouring openssl@1.1--1.1.1l.arm64_big_sur.bottle.tar.gz
==> Regenerating CA certificate bundle from keychain, this may take a while...
🍺  /opt/homebrew/Cellar/openssl@1.1/1.1.1l: 8,073 files, 18MB
==> Installing ruby dependency: readline
==> Pouring readline--8.1.arm64_big_sur.bottle.tar.gz
🍺  /opt/homebrew/Cellar/readline/8.1: 48 files, 1.7MB
==> Installing ruby
==> Pouring ruby--3.0.2.arm64_big_sur.bottle.tar.gz
==> Caveats
By default, binaries installed by gem will be placed into:
  /opt/homebrew/lib/ruby/gems/3.0.0/bin

You may want to add this to your PATH.

ruby is keg-only, which means it was not symlinked into /opt/homebrew,
because macOS already provides this software and installing another version in
parallel can cause all kinds of trouble.

If you need to have ruby first in your PATH, run:
  echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc

For compilers to find ruby you may need to set:
  export LDFLAGS="-L/opt/homebrew/opt/ruby/lib"
  export CPPFLAGS="-I/opt/homebrew/opt/ruby/include"
lihan@LideMacBook-Air ~ % echo 'export PATH="/opt/homebrew/opt/ruby/bin:$PATH"' >> ~/.zshrc

lihan@LideMacBook-Air ~ % source ~/.zshrc

lihan@LideMacBook-Air ~ % ruby -v

ruby 3.0.2p107 (2021-07-07 revision 0db68f0233) [arm64-darwin20]
lihan@LideMacBook-Air ~ % 
```

## 下载Jekyll

```text
lihan@LideMacBook-Air ~ % gem install --user-install bundler jekyll

Fetching bundler-2.2.30.gem
WARNING:  You don't have /Users/lihan/.gem/ruby/3.0.0/bin in your PATH,
	  gem executables will not run.
lihan@LideMacBook-Air ~ % 
```

增加路径，`jekyll new`增加一个文件夹作为blog文件夹，Jekyll会对文件夹做一些初始化。

```text
lihan@LideMacBook-Air Documents % echo 'export PATH="$HOME/.gem/ruby/3.0.0/bin:$PATH"' >> ~/.zshrc

lihan@LideMacBook-Air Documents % source ~/.zshrc                                                 
lihan@LideMacBook-Air Documents % jekyll new Blog  

Running bundle install in /Users/lihan/Documents/Blog... 
  Bundler: Fetching gem metadata from https://rubygems.org/..........
  Bundler: Resolving dependencies...
  Bundler: Using public_suffix 4.0.6
  Bundler: Using bundler 2.2.30
  Bundler: Using colorator 1.1.0
  Bundler: Using concurrent-ruby 1.1.9
  Bundler: Using eventmachine 1.2.7
  Bundler: Using http_parser.rb 0.6.0
  Bundler: Using ffi 1.15.4
  Bundler: Using forwardable-extended 2.6.0
  Bundler: Using rb-fsevent 0.11.0
  Bundler: Using mercenary 0.4.0
  Bundler: Using rexml 3.2.5
  Bundler: Using liquid 4.0.3
  Bundler: Using rouge 3.26.1
  Bundler: Using addressable 2.8.0
  Bundler: Using safe_yaml 1.0.5
  Bundler: Using unicode-display_width 1.8.0
  Bundler: Using sassc 2.4.0
  Bundler: Using rb-inotify 0.10.1
  Bundler: Using kramdown 2.3.1
  Bundler: Using pathutil 0.16.2
  Bundler: Using i18n 1.8.10
  Bundler: Using em-websocket 0.5.2
  Bundler: Using listen 3.7.0
  Bundler: Using terminal-table 2.0.0
  Bundler: Using jekyll-sass-converter 2.1.0
  Bundler: Using jekyll-watch 2.2.1
  Bundler: Using kramdown-parser-gfm 1.1.0
  Bundler: Using jekyll 4.2.1
  Bundler: Fetching jekyll-feed 0.15.1
  Bundler: Fetching jekyll-seo-tag 2.7.1
  Bundler: Installing jekyll-feed 0.15.1
  Bundler: Installing jekyll-seo-tag 2.7.1
  Bundler: Fetching minima 2.5.1
  Bundler: Installing minima 2.5.1
  Bundler: Bundle complete! 6 Gemfile dependencies, 31 gems now installed.
  Bundler: Use `bundle info [gemname]` to see where a bundled gem is installed.
New jekyll site installed in /Users/lihan/Documents/Blog. 

```

尝试`jekyll serve`在本地搭建，如果报错是因为`ruby`版本太高，有些包不包含，手动添加就行(如`bundle add webrick`)，运行成功之后可以在[这里](http://127.0.0.1:4000/)看到初始页面。

```text
lihan@LideMacBook-Air Blog % jekyll serve


                    ------------------------------------------------
      Jekyll 4.2.1   Please append `--trace` to the `serve` command 
                     for any additional information or backtrace. 
                    ------------------------------------------------
/Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/commands/serve/servlet.rb:3:in `require': cannot load such file -- webrick (LoadError)
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/commands/serve/servlet.rb:3:in `<top (required)>'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/commands/serve.rb:179:in `require_relative'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/commands/serve.rb:179:in `setup'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/commands/serve.rb:100:in `process'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/command.rb:91:in `block in process_with_graceful_fail'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/command.rb:91:in `each'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/command.rb:91:in `process_with_graceful_fail'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/lib/jekyll/commands/serve.rb:86:in `block (2 levels) in init_with_program'
	from /Users/lihan/.gem/ruby/3.0.0/gems/mercenary-0.4.0/lib/mercenary/command.rb:221:in `block in execute'
	from /Users/lihan/.gem/ruby/3.0.0/gems/mercenary-0.4.0/lib/mercenary/command.rb:221:in `each'
	from /Users/lihan/.gem/ruby/3.0.0/gems/mercenary-0.4.0/lib/mercenary/command.rb:221:in `execute'
	from /Users/lihan/.gem/ruby/3.0.0/gems/mercenary-0.4.0/lib/mercenary/program.rb:44:in `go'
	from /Users/lihan/.gem/ruby/3.0.0/gems/mercenary-0.4.0/lib/mercenary.rb:21:in `program'
	from /Users/lihan/.gem/ruby/3.0.0/gems/jekyll-4.2.1/exe/jekyll:15:in `<top (required)>'
	from /Users/lihan/.gem/ruby/3.0.0/bin/jekyll:23:in `load'
	from /Users/lihan/.gem/ruby/3.0.0/bin/jekyll:23:in `<main>'
lihan@LideMacBook-Air Blog % bundle add webrick

Fetching gem metadata from https://rubygems.org/.........
Resolving dependencies...
Fetching gem metadata from https://rubygems.org/.........
Resolving dependencies...
Using public_suffix 4.0.6
Using bundler 2.2.30
Using colorator 1.1.0
Using concurrent-ruby 1.1.9
Using eventmachine 1.2.7
Using http_parser.rb 0.6.0
Using ffi 1.15.4
Using rb-fsevent 0.11.0
Using forwardable-extended 2.6.0
Using rexml 3.2.5
Using liquid 4.0.3
Using mercenary 0.4.0
Using safe_yaml 1.0.5
Using unicode-display_width 1.8.0
Fetching webrick 1.7.0
Using rouge 3.26.1
Using addressable 2.8.0
Using kramdown 2.3.1
Using terminal-table 2.0.0
Using kramdown-parser-gfm 1.1.0
Using em-websocket 0.5.2
Using i18n 1.8.10
Using sassc 2.4.0
Using rb-inotify 0.10.1
Using jekyll-sass-converter 2.1.0
Using pathutil 0.16.2
Using listen 3.7.0
Using jekyll-watch 2.2.1
Using jekyll 4.2.1
Using jekyll-feed 0.15.1
Using jekyll-seo-tag 2.7.1
Using minima 2.5.1
Installing webrick 1.7.0
lihan@LideMacBook-Air Blog % jekyll serve      

Configuration file: /Users/lihan/Documents/Blog/_config.yml
            Source: /Users/lihan/Documents/Blog
       Destination: /Users/lihan/Documents/Blog/_site
 Incremental build: disabled. Enable with --incremental
      Generating... 
       Jekyll Feed: Generating feed for posts
                    done in 0.161 seconds.
 Auto-regeneration: enabled for '/Users/lihan/Documents/Blog'
    Server address: http://127.0.0.1:4000/
  Server running... press ctrl-c to stop.
```

接下来就是去挑选主题，[主题网站1](http://jekyllthemes.org)，[主题网站2](http://themes.jekyllrc.org)，[主题网站3](https://jekyllthemes.io)。然后push到自己的github仓库上，关于如何设置github page请看另一篇[How to Build Blog Site on GitHub Using Hexo](https://herloconnell.github.io/blog/How-to-Build-Blog-Site-on-GitHub-Using-Hexo/)

## GitHub Page

**一些常用的操作**

将当前文件夹初始化，建立本地仓库

```text
git init
```

添加所有文件到仓库

```text
git add . 
```

添加多个文件到仓库

```text
git add file1 file2 file3
```

将修改提交到本地仓库

```text
git commit -m "add all file"
```

将修改过的文件进行提交

```text
git commit -a -m "chang some files"
```

关联远程仓库

```text
git remote add origin git@github.com:xxx/xxx
```

从本地仓库推送到远程仓库

```text
git push -u origin master
```

查看修改过的文件

```text
git status
```

查看历史提交

```text
git log
```

恢复历史版本，`-f` 代表force，强制提交

```text
git reset --hard [commit id]
git push -f -u origin master 
```

 `pull `命令等同于先`fetch`再`merge`

```text
git pull origin develop 

git fetch origin develop  
git checkout develop  
git merge origin/develop  
```

使用远程分支覆盖本地

```text
git fetch --all
git reset --hard origin/master (master要修改为对应的分支名)
git pull
```

**.gitignore**

.DS_Store是Mac OS用来存储这个文件夹的显示属性的，被作为一种通用的有关显示设置的元数据（比如图标位置等设置）为Finder、Spotlight用。所以在不经意间就会修改这个文件。而文件共享时为了隐私关系将.DS_Store文件删除比较好，因为其中有一些信息在不经意间泄露出去。

仅针对git的处理最naive的想法就是设置.gitignore文件。

.gitignore文件用于忽略文件，官网介绍在[这里](https://git-scm.com/docs/gitignore)，规范如下：

- 所有空行或者以注释符号 `＃` 开头的行都会被 git 忽略，空行可以为了可读性分隔段落，`#` 表明注释。
- 第一个 `/` 会匹配路径的根目录，举个栗子，”/*.html”会匹配”index.html”，而不是”d/index.html”。
- 通配符 `*` 匹配任意个任意字符，`?` 匹配一个任意字符。需要注意的是通配符不会匹配文件路径中的 `/`，举个栗子，”d/*.html”会匹配”d/index.html”，但不会匹配”d/a/b/c/index.html”。
- 两个连续的星号**有特殊含义：
  - 以 `**/` 开头表示匹配所有的文件夹，例如 `**/test.md` 匹配所有的test.md文件。
  - 以 `/**` 结尾表示匹配文件夹内所有内容，例如 `a/**` 匹配文件夹a中所有内容。
  - 连续星号 `**` 前后分别被 `/` 夹住表示匹配0或者多层文件夹，例如 `a/**/b` 匹配到 `a/b` 、`a/x/b`、`a/x/y/b` 等。
- 前缀 `!` 的模式表示如果前面匹配到被忽略，则重新添加回来。如果匹配到的父文件夹还是忽略状态，该文件还是保持忽略状态。如果路径名第一个字符为 `!` ，则需要在前面增加 `\` 进行转义。

对于一些常用的系统、工程文件的.gitignore文件可以参考[这个网站](https://www.gitignore.io/)进行设置，这里有很多模板。

针对.DS_Store文件，在git工程文件夹中新建.gitignore文件，在文件中设置：

```text
**/.DS_Store
```

.gitignore只能忽略那些原来没有被track的文件，如果某些文件已经被纳入了版本管理中，则修改.gitignore是无效的。解决方法是先把本地缓存删除（改变成未track状态），然后再提交。

```text
git rm -r --cached . 
git add . 
git commit -m 'update .gitignore'
```

