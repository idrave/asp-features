from distutils.core import setup

setup(name='features',
      version='1.0',
      packages=['features', 'features.sample'],
      package_data={'features' : ['*.lp']}
     )