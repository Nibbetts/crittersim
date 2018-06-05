import pyglet
image = pyglet.image.load('resources/bacteria_white.png')
image2 = pyglet.image.load('resources/bacteria_blue.png')
image.anchor_x = 16
image.anchor_y = 32
ball = pyglet.sprite.Sprite(image,0,0)
ball2 = pyglet.sprite.Sprite(image2,0,0)
#ball.scale = 1.5
#ball.rotation = 45
window = pyglet.window.Window()
@window.event
def on_draw():
    ball.rotation += 1
    ball2.rotation -= 2
    ball.draw()
    ball2.draw()
    print(ball.rotation)

pyglet.app.run()
