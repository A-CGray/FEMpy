build: build-mkdocs

docker-build: docker-build-mkdocs

run: run-mkdocs

publish: publish-mkdocs

clean:
	rm -rf docs/site

build-docker:
	docker build -t fempy .




docker-run-mkdocs:
	docker run --rm -it -v `pwd`/docs:/data -p 8000:8000 fempy mkdocs serve -a 0.0.0.0:8000

docker-build-mkdocs: clean
	docker run --rm -it -v `pwd`/docs:/data fempy mkdocs build

docker-publish-mkdocs:
	docker run --rm -it -v `pwd`:/data -v "$(HOME)/.ssh":/root/.ssh fempy mkdocs gh-deploy -f docs/mkdocs.yml

run-mkdocs:
	mkdocs serve -f docs/mkdocs.yml -a 0.0.0.0:8000

build-mkdocs: clean
	mkdocs build -f docs/mkdocs.yml

publish-mkdocs:
	mkdocs gh-deploy -f docs/mkdocs.yml


