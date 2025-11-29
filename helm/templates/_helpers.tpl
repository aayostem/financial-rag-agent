{{/*
Expand the name of the chart.
*/}}
{{- define "financial-rag-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "financial-rag-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "financial-rag-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "financial-rag-agent.labels" -}}
helm.sh/chart: {{ include "financial-rag-agent.chart" . }}
{{ include "financial-rag-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "financial-rag-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "financial-rag-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "financial-rag-agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "financial-rag-agent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create image name
*/}}
{{- define "financial-rag-agent.image" -}}
{{- printf "%s:%s" .Values.app.image.repository .Values.app.image.tag }}
{{- end }}

{{/*
Create environment string
*/}}
{{- define "financial-rag-agent.env" -}}
{{- range .Values.app.env }}
- name: {{ .name }}
  value: {{ .value | quote }}
{{- end }}
{{- end }}

{{/*
Create volume mounts
*/}}
{{- define "financial-rag-agent.volumeMounts" -}}
{{- range .Values.app.volumeMounts }}
- name: {{ .name }}
  mountPath: {{ .mountPath }}
{{- end }}
{{- end }}

{{/*
Create volumes
*/}}
{{- define "financial-rag-agent.volumes" -}}
{{- range .Values.app.volumes }}
- name: {{ .name }}
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}