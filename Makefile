#
# Makefile inspired by CookieCutter Data Science http://drivendata.github.io/cookiecutter-data-science/
# Adapted by Niclas Simmler based on best practices from previous projects.
#

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROFILE = default
PROJECT_NAME = ser
PYTHON_INTERPRETER = python3
DOCKER=docker
FLAGS=
DOCKERFILE=Dockerfile
IMAGE_NAME=$(PROJECT_NAME)-image
CONTAINER_NAME=$(PROJECT_NAME)-container

PWD=$(shell pwd)
export JUPYTER_HOST_PORT=8888
export JUPYTER_CONTAINER_PORT=8888
export FLASK_HOST_PORT=5000
export FLASK_CONTAINER_PORT=5000

define START_DOCKER_CONTAINER
if [ `$(DOCKER) inspect -f {{.State.Running}} $(CONTAINER_NAME)` = "false" ] ; then
        $(DOCKER) start $(CONTAINER_NAME)
fi
endef

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Init everything
init: clean-docker init-docker create-container start-container

## Initialize the Docker Image
init-docker: 
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

## Create the Docker Container
create-container: 
	$(DOCKER) run -t -i -d $(FLAGS)-v $(PWD):/work -p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) -p $(FLASK_HOST_PORT):$(FLASK_CONTAINER_PORT) --name $(CONTAINER_NAME) $(IMAGE_NAME)

## Start the Docker Container
start-container: 
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "Launched $(CONTAINER_NAME)..."
	$(DOCKER) attach $(CONTAINER_NAME)

## Attach to the running container
attach: start-container

## Starts the SER application (jupyter and flask api)
boot-app:
	-screen -dmS API $(PYTHON_INTERPRETER) src/api/main.py
	-screen -dmS JUPYTER jupyter-lab --allow-root --ip=0.0.0.0 --port=${JUPYTER_CONTAINER_PORT}
	-@echo ">>>>>>>>>>>>>>>> Started the API screen! Reattach using screen -r API"
	-@echo ">>>>>>>>>>>>>>>> Started the JUPTYER screen! Reattach using screen -r JUPYTER"
	-@echo ">>>>>>>>>>>>>>>> Attach to the JUPYTER screen to get the auth token! Detach screens using CTRL+A, D"
	-/bin/bash

## Remove the Docker Container
clean-docker: clean-container 

## Remove all Docker related data (image and container). Warning: There will be dragons!
clean-docker-full-only-in-emergency: clean-container clean-image 

## Remove the Docker Container
clean-container: 
	-$(DOCKER) stop $(CONTAINER_NAME)
	-$(DOCKER) rm $(CONTAINER_NAME)

## Remove the Docker Image
clean-image: 
	-$(DOCKER) image rm $(IMAGE_NAME)

## Fix all installed python modules to requirements.txt
pip-freeze:
	pip freeze > ./requirements.txt

## Updates the environment using using the current requirements.txt
update-env: 
	pip install -r ./requirements.txt --upgrade

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
