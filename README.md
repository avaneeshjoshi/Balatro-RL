<!-- LOGO -->
<br />
<h1>
<p align="center">
  <img src="./assets/balatro.png" alt="Logo" width="140" height="140">
  <br>Balatro RL Agent
</h1>
  <p align="center">
    A Reinforcement Learning Agent that learns to play Balatro using deep learning.
    <br />
    </p>
</p>
<p align="center">
  <a href="#about-the-project">About The Project</a> •
  <a href="#usage">Overview</a> •
  <a href="#examples">How It Works</a> •
  <a href="#credits">Credits</a>
</p>  

<table>
<tr>
<td>

This is a _Reinforcement Learning agent_ that attempts to play **Balatro** through Deep Learning. The agent utilizes the **Lovely Injector** to inject into the **Love2D** engine-based game to read different states. (Will add more info as I learn more about the architecture of the agent)

</td>
</tr>
</table>

## About The Project

## Overview

## How It Works
- Initial Setup:

I am using the steam version of the game so follow the Lovely Injector instructions first. After that set the launch options in steam as --dump-all.

Essentially what this does is it uses the Lovely Injector to dump all the game files and then we can search them to see how game info is handled. To find your dump folder on **Windows** go to _AppData\Roaming\Balatro\Mods\lovely\dump_. Since the dump contains the game files we can just open this into VS Code and do _Ctrl+Shift+F_ or _Cmd+Shift+F_ to search through all the game files at once to see stuff like: 1. How score is handled, 2. What happens when you play a hand, etc.

Once you have openned the **dump** folder in VS Code, you will see a bunch of Lua files. These are all the important game files that show how logic is handled.

We will make a Symlink between the bridge folder here and the bridge folder in the Lovely Mods directory.

## Credits
- Credit to [@ethangreen-dev](https://github.com/ethangreen-dev/lovely-injector) for the Love2D Injector code.

## License

This project is open source and available under the MIT License.
